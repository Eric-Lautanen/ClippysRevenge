#!/usr/bin/env python3
"""
Synthetic multi-turn conversational dataset generator — LM Studio headless edition.
Generates 25 convos per seed word, 5-15 turns each, various flavors/personas.
Output: JSONL files (50 MB max) in /conversational_dataset
Schema: {seed_word, flavor, persona1, persona2, messages}

LM Studio REST API v1 (≥ 0.3.6):
  GET  /api/v1/models           – list + check loaded instances
  POST /api/v1/models/load      – load model by key
  POST /api/v1/models/unload    – unload by instance_id
  POST /v1/chat/completions     – OpenAI-compat, streaming
"""

import argparse
import json
import os
import re
import signal
import sys
import random
import hashlib
import logging
import time
import requests
from pathlib import Path

# ── config (CLI overrides these at runtime) ───────────────────────────────────
WORDS_FILE      = "words.txt"
OUTPUT_DIR      = "./conversational_dataset"           # hardcoded, always here
MAX_FILE_BYTES  = 50 * 1024 * 1024                    # 50 MB per part file
CONVOS_PER_WORD = 25
TOP_N_WORDS     = 10_000
MIN_TURNS       = 5
MAX_TURNS       = 15
TEMPERATURE     = 0.4
MAX_TOKENS      = 4096
RANDOM_SEED     = 42
LMS_HOST        = "http://localhost:1234"             # LM Studio default port
MODEL_NAME      = "qwen3.5-4b@q3_k_m"  # default; override with --model
BATCH_SIZE      = 50                                  # convos before cooldown
COOLDOWN_SECS   = 60                                  # seconds to cool down between batches
REQUEST_TIMEOUT = 30                                 # per-request timeout in seconds

# ── flavors ───────────────────────────────────────────────────────────────────
FLAVORS = [
    "rant",
    "vent",
    "joke",
    "advice",
    "genuine question",
    "follow-up chain",
    "indifferent shrug",
    "sarcastic poke",
    "nostalgic ramble",
    "heated debate",
    "confused newbie",
    "know-it-all lecture",
    "half-hearted complaint",
    "excited discovery",
    "blunt reality check",
]

# ── personas ──────────────────────────────────────────────────────────────────
PERSONAS = [
    "cheerful dad",
    "pissed mechanic",
    "quiet observer",
    "teen eye-roll",
    "retired nurse",
    "conspiracy hobbyist",
    "burnt-out teacher",
    "overconfident intern",
    "dry-wit bartender",
    "anxious first-timer",
]

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_words(path: str, top_n: int) -> list[str]:
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                words.append(w)
            if len(words) >= top_n:
                break
    log.info("Loaded %d words from %s", len(words), path)
    return words


# ── shutdown flag (set by signal handlers) ────────────────────────────────────
_shutdown = False

def _handle_signal(signum, frame):
    global _shutdown
    log.warning("Signal %s received — finishing current convo then shutting down...", signum)
    _shutdown = True


# ── LM Studio model manager ───────────────────────────────────────────────────

class LMStudioManager:
    """Manages model load/unload lifecycle against LM Studio's v1 REST API."""

    def __init__(self, host: str, model_key: str, api_token: str | None = None):
        self.host        = host.rstrip("/")
        self.model_key   = model_key
        self.instance_id: str | None = None
        self.session     = requests.Session()
        if api_token:
            self.session.headers.update({"Authorization": f"Bearer {api_token}"})
        self.session.headers.update({"Content-Type": "application/json"})

    # ── internal helpers ──────────────────────────────────────────────────────

    def _get(self, path: str, **kwargs) -> requests.Response:
        return self.session.get(f"{self.host}{path}", timeout=REQUEST_TIMEOUT, **kwargs)

    def _post(self, path: str, **kwargs) -> requests.Response:
        return self.session.post(f"{self.host}{path}", timeout=REQUEST_TIMEOUT, **kwargs)

    # ── server health ─────────────────────────────────────────────────────────

    def wait_for_server(self, retries: int = 12, delay: float = 5.0) -> bool:
        """Block until LM Studio server is reachable or give up."""
        for attempt in range(1, retries + 1):
            try:
                r = self._get("/api/v1/models")
                if r.status_code == 200:
                    log.info("LM Studio server reachable at %s", self.host)
                    return True
            except requests.exceptions.ConnectionError:
                pass
            log.info("Waiting for LM Studio server... (%d/%d)", attempt, retries)
            time.sleep(delay)
        log.error("LM Studio server not reachable after %d attempts.", retries)
        return False

    # ── model list ────────────────────────────────────────────────────────────

    def list_models(self) -> list[dict]:
        """Return raw model list from /api/v1/models."""
        try:
            r = self._get("/api/v1/models")
            r.raise_for_status()
            return r.json().get("models", [])
        except Exception as e:
            log.error("list_models failed: %s", e)
            return []

    def loaded_instances(self) -> list[tuple[str, str]]:
        """Return [(model_key, instance_id), ...] for all currently loaded models."""
        result = []
        for m in self.list_models():
            for inst in m.get("loaded_instances", []):
                result.append((m["key"], inst["id"]))
        return result

    # ── unload ────────────────────────────────────────────────────────────────

    def unload(self, instance_id: str) -> bool:
        """Unload a specific instance by ID. Returns True on success."""
        try:
            r = self._post("/api/v1/models/unload", json={"instance_id": instance_id})
            if r.status_code == 200:
                log.info("Unloaded instance: %s", instance_id)
                if self.instance_id == instance_id:
                    self.instance_id = None
                return True
            log.warning("Unload returned %d for %s: %s", r.status_code, instance_id, r.text[:200])
            return False
        except Exception as e:
            log.error("unload(%s) failed: %s", instance_id, e)
            return False

    def unload_all_others(self) -> None:
        """Unload any loaded model that isn't ours."""
        for key, iid in self.loaded_instances():
            if iid != self.instance_id:
                log.info("Evicting foreign model: key=%s instance=%s", key, iid)
                self.unload(iid)

    def unload_ours(self) -> None:
        """Unload our model if it's currently loaded."""
        if self.instance_id:
            self.unload(self.instance_id)

    # ── load ──────────────────────────────────────────────────────────────────

    def load(self, context_length: int = 4096, flash_attention: bool = True) -> bool:
        """Load our model. Evicts any conflicting loaded model first."""
        # kick out anything already loaded
        self.unload_all_others()

        # check if we're already loaded
        for key, iid in self.loaded_instances():
            if key == self.model_key:
                log.info("Model '%s' already loaded (instance=%s)", self.model_key, iid)
                self.instance_id = iid
                return True

        log.info("Loading model '%s'...", self.model_key)
        payload = {
            "model":            self.model_key,
            "context_length":   context_length,
            "flash_attention":  flash_attention,
            "echo_load_config": True,
        }
        try:
            r = self._post("/api/v1/models/load", json=payload)
            if r.status_code == 200:
                data = r.json()
                self.instance_id = data.get("instance_id", self.model_key)
                load_time = data.get("load_time_seconds", "?")
                log.info("Model loaded in %.1fs  instance_id=%s", load_time, self.instance_id)
                return True
            log.error("Load failed (%d): %s", r.status_code, r.text[:400])
            return False
        except Exception as e:
            log.error("load() exception: %s", e)
            return False

    # ── cycle: unload → sleep → reload ───────────────────────────────────────

    def cooldown_cycle(self, cooldown_secs: int) -> bool:
        """Unload model, sleep to let the laptop cool, reload. Returns False if reload fails."""
        log.info("── COOLDOWN: unloading model for %ds rest ──", cooldown_secs)
        self.unload_ours()
        for remaining in range(cooldown_secs, 0, -10):
            log.info("  cooling down... %ds remaining", remaining)
            time.sleep(min(10, remaining))
            if _shutdown:
                return False
        log.info("── RELOADING model after cooldown ──")
        return self.load()

    # ── inference: streaming chat completion ─────────────────────────────────

    def chat(self, messages: list[dict], temperature: float = TEMPERATURE,
             stream: bool = True) -> str:
        """
        POST /v1/chat/completions (OpenAI-compat).
        Streams tokens to stdout in real time, returns full assembled text.
        Falls back to non-stream if stream=False.
        """
        if self.instance_id is None:
            log.error("chat() called but no model is loaded")
            return ""

        payload = {
            "model":       self.instance_id,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  MAX_TOKENS,
            "stream":      stream,
        }

        try:
            if stream:
                return self._stream_chat(payload)
            else:
                r = self._post("/v1/chat/completions", json=payload)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            log.error("chat() timed out after %ds", REQUEST_TIMEOUT)
            return ""
        except Exception as e:
            log.error("chat() failed: %s", e)
            return ""

    def _stream_chat(self, payload: dict) -> str:
        """Stream SSE tokens, print to terminal, return full string."""
        collected = []
        try:
            with self.session.post(
                f"{self.host}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=REQUEST_TIMEOUT,
            ) as resp:
                resp.raise_for_status()
                print("\n\033[90m[stream] ", end="", flush=True)
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        delta = json.loads(chunk)
                        token = (
                            delta.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if token:
                            print(token, end="", flush=True)
                            collected.append(token)
                    except json.JSONDecodeError:
                        pass
                print("\033[0m", flush=True)   # reset colour
        except Exception as e:
            log.error("_stream_chat failed mid-stream: %s", e)
        return "".join(collected).strip()


# module-level client (set in main after CLI parse)
lms: LMStudioManager | None = None


# ── prompt builders ───────────────────────────────────────────────────────────

FLAVOR_HINTS = {
    "rant": "speaker1 is ranting, frustrated, going off",
    "vent": "speaker1 is venting about something personal or annoying",
    "joke": "speaker1 opens with a joke or funny angle",
    "advice": "speaker1 is asking for practical advice",
    "genuine question": "speaker1 has a sincere, curious question",
    "follow-up chain": "each reply spawns a deeper follow-up question",
    "indifferent shrug": "speaker1 doesn't really care, low energy, whatever",
    "sarcastic poke": "speaker1 is being sarcastic or poking fun",
    "nostalgic ramble": "speaker1 is reminiscing, going down memory lane",
    "heated debate": "both speakers disagree and push back on each other",
    "confused newbie": "speaker1 is lost/confused and asking basic stuff",
    "know-it-all lecture": "speaker1 acts like the expert and talks at speaker2",
    "half-hearted complaint": "speaker1 complains but doesn't actually care that much",
    "excited discovery": "speaker1 just found out something and is hyped",
    "blunt reality check": "speaker2 keeps cutting through the BS with harsh truths",
    "quiet confession": "speaker1 whispers something real, no drama",
    "defensive snap": "speaker1 bites back, gets prickly fast",
    "random tangent": "speaker1 drifts off topic, forgets the point",
    "mock sympathy": "speaker1 fake-cares, eye-roll heavy",
    "angry whisper": "speaker1 low-key furious, hisses words",
    "drunk ramble": "speaker1 slurs, no sense, laughs at nothing",
    "wise old shrug": "speaker1 drops truth like it's obvious",
    "teen tantrum": "speaker1 stomps, 'this sucks!' vibes",
    "conspiracy whisper": "speaker1 leans in, 'they're watching'",
    "bored dismissal": "speaker1 waves it off, 'whatever'",
    "over-the-top hype": "speaker1 yells, 'this is life-changing!'",
    "silent stare-down": "speaker1 says nothing, just glares",
    "bitter laugh": "speaker1 chuckles dark, life's a joke",
    "desperate plea": "speaker1 begs, 'just tell me'",
    "monk zen": "speaker1 calm, drops one-line wisdom"
}

PERSONA_HINTS = {
    "cheerful dad": "talks like a friendly middle-aged dad, uses 'buddy', mild jokes",
    "pissed mechanic": "short sentences, profanity OK, no patience for dumb questions",
    "quiet observer": "measured, thoughtful, says less but means more",
    "teen eye-roll": "uses 'like', 'whatever', short replies, slightly annoyed",
    "retired nurse": "practical, seen-it-all, matter-of-fact, occasional dark humor",
    "conspiracy hobbyist": "connects everything to something bigger, 'they' don't want you to know",
    "burnt-out teacher": "exhausted, still tries to explain things, mild sarcasm",
    "overconfident intern": "acts like they know everything, often wrong but never doubts",
    "dry-wit bartender": "short quips, seen too much, deadpan",
    "anxious first-timer": "lots of questions, second-guesses everything, apologetic",
    "cynical barista": "sarcastic, coffee-obsessed, judges your order silently",
    "grumpy grandpa": "complains about 'kids these days', grunts a lot",
    "hyper kid": "talks fast, no filter, 'that's so cool!' every sentence",
    "wise monk": "calm, zen, drops one-line truths like bombs",
    "flirty coworker": "winks, teases, keeps it light but pushes boundaries",
    "paranoid neighbor": "whispers, checks windows, 'you hear that?'",
    "lazy gamer": "short answers, references memes, 'brb afk'",
    "corporate drone": "uses buzzwords, 'synergy', hates meetings",
    "punk rocker": "loud, swears, hates authority, 'fuck the system'",
    "bookworm nerd": "quotes authors, corrects grammar, shy but passionate",
    "gym bro": "bro-y, flexes, 'gains', protein shakes",
    "hippie mom": "peaceful, 'vibes', essential oils, no judgment",
    "bitter ex": "passive-aggressive, brings up old fights",
    "optimistic salesman": "hypes everything, 'this'll change your life'",
    "lonely trucker": "long stories, road tales, misses family",
    "sassy grandma": "sharp tongue, 'back in my day', winks",
    "doomer": "everything's doomed, sighs, 'why bother?'",
    "tech bro": "crypto, AI, 'disrupt', talks over you",
    "shy artist": "mumbles, describes colors, avoids eye contact",
    "drunk uncle": "slurs, laughs too loud, 'remember that time?'"
}


def build_generation_prompt(seed_word: str, flavor: str, p1: str, p2: str,
                             turn_count: int) -> list[dict]:
    hint_flavor = FLAVOR_HINTS.get(flavor, flavor)
    hint_p1     = PERSONA_HINTS.get(p1, p1)
    hint_p2     = PERSONA_HINTS.get(p2, p2)

    system = (
        "You are a dataset generator. Your job is to write a realistic, raw, "
        "unfiltered multi-turn conversation between two people. "
        "No user/assistant labels. Use speaker1 and speaker2 only. "
        "Do NOT summarize or explain—write ONLY the dialogue. "
        "English only. Natural, honest, not performatively helpful. "
        "No sycophancy. Contractions, slang, interruptions fine. "
        "Output ONLY valid JSON: a list of objects, each with 'speaker' and 'text'. "
        "No markdown fences, no extra keys, no commentary outside the JSON array."
    )

    user = (
        f"Seed word: \"{seed_word}\"\n"
        f"Flavor: {flavor} — {hint_flavor}\n"
        f"speaker1 persona: {p1} — {hint_p1}\n"
        f"speaker2 persona: {p2} — {hint_p2}\n"
        f"Turn count: exactly {turn_count} turns (speaker1 starts, alternates).\n\n"
        "The conversation must naturally use or relate to the seed word. "
        "Make it feel real—people talk past each other, get distracted, trail off. "
        "End it naturally (don't wrap up too neatly). Write the JSON array now."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


def build_prompt_list_prompt(seed_word: str, flavors_sample: list[str],
                              personas_sample: list[str]) -> list[dict]:
    """Ask the LLM for 25 varied opening lines for speaker1."""
    system = (
        "You are a prompt generator for a conversational dataset. "
        "Output ONLY a JSON array of 25 short strings. "
        "Each string is a natural opening line or question that a real person might say. "
        "No commentary, no markdown, no numbering outside the JSON array."
    )
    user = (
        f"Seed word: \"{seed_word}\"\n"
        f"Flavors to cover (rotate through): {', '.join(flavors_sample)}\n"
        f"Personas to cover (rotate through): {', '.join(personas_sample)}\n\n"
        "Generate 25 distinct, authentic opening lines—vents, rants, questions, jokes, "
        "sarcasm, confusion, excitement—that a real person might say when the topic is "
        f"'{seed_word}'. Vary length and tone. Output the JSON array only."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


# ── robust JSON parser / fixer ────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` markdown fences."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")
    return text.strip()

def _extract_array(text: str) -> str | None:
    """Pull out the outermost [...] block."""
    start = text.find("[")
    end   = text.rfind("]")
    if start != -1 and end > start:
        return text[start:end + 1]
    return None

def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ]."""
    return re.sub(r",\s*([}\]])", r"\1", text)

def _fix_single_quotes(text: str) -> str:
    """Replace single-quoted strings with double-quoted (naive but catches most cases)."""
    # Only replace unescaped single quotes used as delimiters
    return re.sub(r"(?<![\\])'", '"', text)

def _fix_unquoted_keys(text: str) -> str:
    """Quote bare object keys like  speaker: → "speaker":"""
    return re.sub(r'(?<=[{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', r'"\1":', text)

def _fix_newlines_in_strings(text: str) -> str:
    """Replace literal newlines inside JSON string values with \\n."""
    # Replace newlines that appear between quotes
    result = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == "\\":
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string and ch == "\n":
            result.append("\\n")
            continue
        result.append(ch)
    return "".join(result)

def _truncate_to_last_complete_object(text: str) -> str:
    """If JSON is truncated mid-array, trim to the last complete {...} and close the array."""
    last_close = text.rfind("}")
    if last_close == -1:
        return text
    return text[:last_close + 1] + "]"

def fix_and_parse_json(raw: str) -> list | None:
    """
    Multi-strategy JSON fixer. Tries progressively more aggressive repairs.
    Returns a list on success, None on total failure.
    """
    if not raw or not raw.strip():
        return None

    candidates = [raw]

    # strategy 1: strip fences
    s1 = _strip_fences(raw)
    candidates.append(s1)

    # strategy 2: extract array
    s2 = _extract_array(s1) or _extract_array(raw)
    if s2:
        candidates.append(s2)

    # strategy 3: fix trailing commas
    if s2:
        candidates.append(_fix_trailing_commas(s2))

    # strategy 4: fix single quotes
    if s2:
        candidates.append(_fix_single_quotes(_fix_trailing_commas(s2)))

    # strategy 5: fix unquoted keys + trailing commas
    if s2:
        candidates.append(_fix_unquoted_keys(_fix_trailing_commas(s2)))

    # strategy 6: fix newlines inside strings
    if s2:
        candidates.append(_fix_newlines_in_strings(_fix_trailing_commas(s2)))

    # strategy 7: all fixes stacked
    if s2:
        fully_fixed = _fix_newlines_in_strings(
            _fix_trailing_commas(
                _fix_unquoted_keys(
                    _fix_single_quotes(s2)
                )
            )
        )
        candidates.append(fully_fixed)

    # strategy 8: truncate to last complete object (handles mid-stream cutoff)
    if s2:
        candidates.append(_truncate_to_last_complete_object(
            _fix_trailing_commas(s2)
        ))

    for attempt in candidates:
        if not attempt:
            continue
        try:
            result = json.loads(attempt)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            continue

    return None


def validate_messages(msgs: list, turn_count: int) -> bool:
    if not isinstance(msgs, list) or len(msgs) < MIN_TURNS:
        return False
    for m in msgs:
        if not isinstance(m, dict):
            return False
        if "speaker" not in m or "text" not in m:
            return False
        if m["speaker"] not in ("speaker1", "speaker2"):
            return False
        if not isinstance(m["text"], str) or not m["text"].strip():
            return False
    return True


# ── file management ───────────────────────────────────────────────────────────

class OutputManager:
    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.part_index  = 1
        self.current_fh  = None
        self.current_path = None
        self.current_size = 0
        self.seen_hashes  = set()
        self._open_new_file()

    def _open_new_file(self):
        if self.current_fh:
            self.current_fh.close()
        self.current_path = self.output_dir / f"layer1-part{self.part_index:03d}.jsonl"
        self.current_fh   = open(self.current_path, "a", encoding="utf-8")
        self.current_size = self.current_path.stat().st_size if self.current_path.exists() else 0
        log.info("Writing to %s", self.current_path)

    def write(self, record: dict) -> bool:
        """Returns True if written, False if duplicate."""
        convo_text = "".join(m["text"] for m in record["messages"])
        h = md5(convo_text)
        if h in self.seen_hashes:
            return False
        self.seen_hashes.add(h)

        line = json.dumps(record, ensure_ascii=False) + "\n"
        encoded = line.encode("utf-8")

        if self.current_size + len(encoded) > MAX_FILE_BYTES:
            self.part_index += 1
            self._open_new_file()

        self.current_fh.write(line)
        self.current_fh.flush()
        self.current_size += len(encoded)
        return True

    def close(self):
        if self.current_fh:
            self.current_fh.close()


# ── core generation ───────────────────────────────────────────────────────────

def generate_opening_prompts(seed_word: str, rng: random.Random) -> list[str]:
    """Generate 25 opening lines for a given seed word."""
    flavors_sample  = rng.sample(FLAVORS,   min(8, len(FLAVORS)))
    personas_sample = rng.sample(PERSONAS,  min(6, len(PERSONAS)))
    prompt = build_prompt_list_prompt(seed_word, flavors_sample, personas_sample)
    raw    = lms.chat(prompt, temperature=0.7)
    parsed = fix_and_parse_json(raw)

    if parsed and isinstance(parsed, list) and len(parsed) >= 5:
        lines = [str(x).strip() for x in parsed if isinstance(x, str) and x.strip()]
        if len(lines) >= 5:
            return lines[:25]

    log.warning("Prompt list parse failed for '%s', using fallback stubs", seed_word)
    fallback_templates = [
        f"So about {seed_word}—what's the actual deal with that?",
        f"Can someone explain {seed_word} like I'm not an idiot?",
        f"I'm so tired of hearing about {seed_word}.",
        f"Okay, {seed_word}. Thoughts?",
        f"Nobody talks about {seed_word} enough.",
        f"Does anyone actually understand {seed_word}?",
        f"I just looked up {seed_word} and now I have more questions.",
        f"Hot take: {seed_word} is overrated.",
        f"Why is {seed_word} suddenly everywhere?",
        f"My dad won't stop talking about {seed_word}.",
        f"Is it just me or is {seed_word} kind of a scam?",
        f"Alright, change my mind about {seed_word}.",
        f"{seed_word}. That's it. That's the message.",
        f"I made a mistake involving {seed_word} and I'm not over it.",
        f"Quick question about {seed_word}—anyone got five minutes?",
        f"Tell me something real about {seed_word}.",
        f"Every time I think I get {seed_word}, I don't.",
        f"I just had the worst experience with {seed_word}.",
        f"Unpopular opinion about {seed_word}.",
        f"Okay so {seed_word}—am I wrong about this?",
        f"Someone at work brought up {seed_word} and I froze.",
        f"Why does {seed_word} stress me out so much?",
        f"I've been thinking about {seed_word} more than I should.",
        f"Is {seed_word} actually useful or just hype?",
        f"Quick rant about {seed_word}, bear with me.",
    ]
    rng.shuffle(fallback_templates)
    return fallback_templates[:25]


def generate_convo(seed_word: str, opening: str, flavor: str,
                   p1: str, p2: str, rng: random.Random) -> list[dict] | None:
    turn_count = rng.randint(MIN_TURNS, MAX_TURNS)
    # inject the opening line as speaker1's first turn in the system hint
    prompt = build_generation_prompt(seed_word, flavor, p1, p2, turn_count)
    # append the opening line to the user message so the model starts from it
    prompt[-1]["content"] += (
        f"\n\nspeaker1's first line must be (or closely based on): \"{opening}\""
    )
    raw    = lms.chat(prompt, temperature=TEMPERATURE)
    parsed = fix_and_parse_json(raw)
    if parsed and validate_messages(parsed, turn_count):
        return parsed
    log.debug("Convo parse/validate failed for '%s' / '%s'", seed_word, flavor)
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic conversational dataset via LM Studio headless server."
    )
    p.add_argument("--model",      default=MODEL_NAME,
                   help="LM Studio model key, e.g. 'lmstudio-community/gemma-3-1b-it-qat'")
    p.add_argument("--host",       default=LMS_HOST,
                   help=f"LM Studio server base URL (default: {LMS_HOST})")
    p.add_argument("--token",      default=None,
                   help="LM Studio API bearer token (optional, from Developer Settings)")
    p.add_argument("--words",      default=WORDS_FILE,
                   help=f"Path to word list file (default: {WORDS_FILE})")
    p.add_argument("--top-n",      type=int, default=TOP_N_WORDS,
                   help=f"Top N words to process (default: {TOP_N_WORDS})")
    p.add_argument("--convos",     type=int, default=CONVOS_PER_WORD,
                   help=f"Conversations per seed word (default: {CONVOS_PER_WORD})")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help=f"Convos to generate before a cooldown pause (default: {BATCH_SIZE})")
    p.add_argument("--cooldown",   type=int, default=COOLDOWN_SECS,
                   help=f"Cooldown duration in seconds (default: {COOLDOWN_SECS})")
    p.add_argument("--seed",       type=int, default=RANDOM_SEED,
                   help=f"Random seed (default: {RANDOM_SEED})")
    p.add_argument("--temp",       type=float, default=TEMPERATURE,
                   help=f"Inference temperature (default: {TEMPERATURE})")
    p.add_argument("--no-stream",  action="store_true",
                   help="Disable token streaming (faster but no live terminal output)")
    p.add_argument("--context",    type=int, default=4096,
                   help="Model context length to load with (default: 4096)")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global lms, TEMPERATURE

    args = parse_args()

    # apply CLI overrides to module globals used by generator functions
    TEMPERATURE = args.temp

    # wire up signal handlers for graceful exit
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    random.seed(args.seed)
    rng = random.Random(args.seed)

    # ── set up LM Studio client ───────────────────────────────────────────────
    lms = LMStudioManager(host=args.host, model_key=args.model, api_token=args.token)

    if not lms.wait_for_server():
        log.error("Cannot reach LM Studio at %s — is the server running?", args.host)
        sys.exit(1)

    log.info("Checking for already-loaded models...")
    existing = lms.loaded_instances()
    if existing:
        for key, iid in existing:
            if key != args.model:
                log.info("  Found foreign model loaded: key=%s  instance=%s — unloading", key, iid)
                lms.unload(iid)
            else:
                log.info("  Our model already loaded: instance=%s", iid)
                lms.instance_id = iid

    if lms.instance_id is None:
        if not lms.load(context_length=args.context):
            log.error("Failed to load model '%s'. Aborting.", args.model)
            sys.exit(1)

    # ── register atexit unload ────────────────────────────────────────────────
    import atexit
    def _cleanup():
        if lms and lms.instance_id:
            log.info("atexit: unloading model '%s'", lms.instance_id)
            lms.unload_ours()
    atexit.register(_cleanup)

    # ── ensure output dir exists ──────────────────────────────────────────────
    output = OutputManager(OUTPUT_DIR)

    # ── load word list ────────────────────────────────────────────────────────
    words = load_words(args.words, args.top_n)

    total_written  = 0
    total_skipped  = 0
    batch_counter  = 0       # convos since last cooldown
    stream_enabled = not args.no_stream

    # monkey-patch streaming flag into the chat wrapper
    original_chat = lms.chat
    def _chat_with_stream_flag(messages, temperature=TEMPERATURE):
        return original_chat(messages, temperature=temperature, stream=stream_enabled)
    lms.chat = _chat_with_stream_flag

    for word_idx, word in enumerate(words):
        if _shutdown:
            log.info("Shutdown flag set — stopping after current word.")
            break

        log.info("── word %d/%d: '%s' ──", word_idx + 1, len(words), word)
        t_word_start = time.time()

        openings = generate_opening_prompts(word, rng)

        # pad / trim to exactly args.convos
        while len(openings) < args.convos:
            openings.append(f"Let's talk about {word}.")
        openings = openings[:args.convos]

        # pre-assign flavors and persona pairs
        flavors_assigned = [FLAVORS[i % len(FLAVORS)] for i in range(args.convos)]
        rng.shuffle(flavors_assigned)

        written_for_word = 0
        for i in range(args.convos):
            if _shutdown:
                break

            # ── cooldown check ────────────────────────────────────────────────
            if batch_counter > 0 and batch_counter % args.batch_size == 0:
                ok = lms.cooldown_cycle(args.cooldown)
                if not ok or _shutdown:
                    log.info("Cooldown interrupted or reload failed — stopping.")
                    break
                # re-apply stream patch after reload
                lms.chat = _chat_with_stream_flag

            flavor  = flavors_assigned[i]
            p1, p2  = rng.sample(PERSONAS, 2)
            opening = openings[i]

            log.info("  [%d/%d] flavor=%-22s  p1=%-22s  p2=%s",
                     i + 1, args.convos, flavor, p1, p2)

            msgs = None
            for attempt in range(3):
                if _shutdown:
                    break
                msgs = generate_convo(word, opening, flavor, p1, p2, rng)
                if msgs:
                    break
                log.debug("  retry %d for word='%s' convo=%d", attempt + 1, word, i)
                time.sleep(1)   # brief pause between retries

            if not msgs:
                log.warning("  gave up on convo %d for '%s'", i, word)
                total_skipped += 1
                batch_counter += 1
                continue

            record = {
                "seed_word": word,
                "flavor":    flavor,
                "persona1":  p1,
                "persona2":  p2,
                "messages":  msgs,
            }

            if output.write(record):
                written_for_word += 1
                total_written    += 1
            else:
                log.debug("  duplicate skipped for '%s'", word)
                total_skipped += 1

            batch_counter += 1

        elapsed = time.time() - t_word_start
        log.info(
            "  done: %d written, %d skipped, %.1fs | total: written=%d skipped=%d",
            written_for_word, args.convos - written_for_word,
            elapsed, total_written, total_skipped,
        )

    # ── wrap up ───────────────────────────────────────────────────────────────
    output.close()
    lms.unload_ours()
    log.info("═══ Finished. Total written: %d | skipped: %d ═══", total_written, total_skipped)


if __name__ == "__main__":
    main()