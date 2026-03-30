"""
generate.py
===========
Interactive inference + automated test suite for the RustSLM pretrained model.

Since this is a raw pretrained model (no SFT yet), it completes text rather
than following instructions. Give it a Rust prefix and it continues it.

Usage:
    python generate.py
    python generate.py --checkpoint C:/llm/runs/run1/ckpt_best.pt
    python generate.py --checkpoint ckpt_best.pt --temperature 0.7 --top_k 40
    python generate.py --test                          # run full automated test suite
    python generate.py --test --category ownership     # run one category only
    python generate.py --bench                         # test suite + save report to file

REPL commands:
    :help          show all commands
    :test          run automated test suite interactively
    :test <cat>    run one category  (ownership / lifetimes / async / traits /
                                      errors / collections / closures / unsafe / misc)
    :bench         run full suite and save timestamped JSON report
    :sample <n>    generate N completions for the next prompt (compare diversity)
    :save          save last output to log file
    :temp  <f>     set temperature      (0.4=focused  0.7=balanced  1.0=creative)
    :topk  <n>     set top-k            (20=tight  40=default  100=loose)
    :max   <n>     set max new tokens
    :rep   <f>     set repetition penalty  (1.0=off  1.3=moderate  2.0=strong)
    :win   <n>     set repetition window   (recent N tokens penalised)
    :settings      show current settings
    :cats          list all test categories
    :quit          exit
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import RustSLM, ModelConfig


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT  = "C:/llm/runs/run1/ckpt_best.pt"
DEFAULT_TOKENIZER   = "C:/llm/tokenizer"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K       = 40
DEFAULT_MAX_NEW     = 256
DEFAULT_REP_PENALTY = 1.3
DEFAULT_REP_WINDOW  = 64
DEFAULT_LOG_DIR     = "C:/llm/inference_logs"


# ---------------------------------------------------------------------------
# Automated test suite
# Covers the major categories in your training data.
# Each entry: (label, prompt)
# ---------------------------------------------------------------------------
TEST_SUITE: dict[str, list[tuple[str, str]]] = {
    "ownership": [
        ("move semantics",
         "fn take_ownership(s: String) -> String {\n    "),
        ("borrow immutable",
         "fn print_length(s: &String) {\n    "),
        ("borrow mutable",
         "fn append_world(s: &mut String) {\n    "),
        ("clone vec",
         "fn clone_and_modify(original: &Vec<i32>) -> Vec<i32> {\n    let mut copy = original.clone();\n    "),
        ("drop scope",
         "fn process_data() {\n    let data = vec![1, 2, 3];\n    {\n        let _ref = &data;\n    }\n    "),
    ],
    "lifetimes": [
        ("basic annotation",
         "fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {\n    "),
        ("struct with ref",
         "#[derive(Debug)]\nstruct Important<'a> {\n    part: &'a str,\n}\n\nimpl<'a> Important<'a> {\n    fn new(text: &'a str) -> Self {\n        "),
        ("static lifetime",
         "fn get_greeting() -> &'static str {\n    "),
        ("lifetime elision",
         "fn first_word(s: &str) -> &str {\n    "),
    ],
    "async": [
        ("tokio main",
         "use tokio;\n\n#[tokio::main]\nasync fn main() {\n    "),
        ("async result",
         "async fn fetch_url(url: &str) -> Result<String, reqwest::Error> {\n    "),
        ("spawn task",
         "use tokio::task;\n\nasync fn run_parallel() {\n    let handle = task::spawn(async {\n        "),
        ("async trait",
         "use async_trait::async_trait;\n\n#[async_trait]\ntrait DataStore {\n    async fn fetch(&self, key: &str) -> Option<String>;\n}\n\nstruct MemStore;\n\n#[async_trait]\nimpl DataStore for MemStore {\n    async fn fetch(&self, key: &str) -> Option<String> {\n        "),
    ],
    "traits": [
        ("basic impl",
         "trait Greet {\n    fn hello(&self) -> String;\n}\n\nstruct Person {\n    name: String,\n}\n\nimpl Greet for Person {\n    fn hello(&self) -> String {\n        "),
        ("default method",
         "trait Summary {\n    fn summarize_author(&self) -> String;\n\n    fn summarize(&self) -> String {\n        "),
        ("display",
         "use std::fmt;\n\nstruct Point {\n    x: f64,\n    y: f64,\n}\n\nimpl fmt::Display for Point {\n    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {\n        "),
        ("iterator",
         "struct Counter {\n    count: u32,\n    max: u32,\n}\n\nimpl Iterator for Counter {\n    type Item = u32;\n\n    fn next(&mut self) -> Option<Self::Item> {\n        "),
    ],
    "errors": [
        ("custom error enum",
         "use std::fmt;\n\n#[derive(Debug)]\nenum AppError {\n    NotFound(String),\n    ParseError(String),\n    IoError(std::io::Error),\n}\n\nimpl fmt::Display for AppError {\n    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {\n        "),
        ("question mark",
         "use std::fs;\nuse std::io;\n\nfn read_username_from_file() -> Result<String, io::Error> {\n    let s = fs::read_to_string(\"hello.txt\")?;\n    "),
        ("from impl",
         "impl From<std::io::Error> for AppError {\n    fn from(e: std::io::Error) -> Self {\n        "),
        ("match result",
         "fn parse_port(s: &str) -> Result<u16, String> {\n    match s.parse::<u16>() {\n        "),
    ],
    "collections": [
        ("hashmap word count",
         "use std::collections::HashMap;\n\nfn word_count(text: &str) -> HashMap<&str, usize> {\n    let mut map = HashMap::new();\n    "),
        ("vec iter chain",
         "fn filter_and_double(nums: &[i32]) -> Vec<i32> {\n    nums.iter()\n        "),
        ("hashset dedup",
         "use std::collections::HashSet;\n\nfn find_duplicates(items: &[i32]) -> Vec<i32> {\n    let mut seen = HashSet::new();\n    "),
        ("btreemap freq",
         "use std::collections::BTreeMap;\n\nfn sorted_frequency(words: &[&str]) -> BTreeMap<&str, usize> {\n    let mut freq = BTreeMap::new();\n    "),
    ],
    "closures": [
        ("map filter collect",
         "fn process_numbers(nums: Vec<i32>) -> Vec<i32> {\n    nums.into_iter()\n        .filter(|&x| x > 0)\n        .map(|x| "),
        ("higher order fn",
         "fn make_adder(x: i32) -> impl Fn(i32) -> i32 {\n    "),
        ("sort by key",
         "fn sort_by_length(mut words: Vec<String>) -> Vec<String> {\n    words.sort_by(|a, b| "),
        ("fold sum",
         "fn sum_of_squares(nums: &[i32]) -> i32 {\n    nums.iter().fold(0, |acc, &x| "),
    ],
    "unsafe": [
        ("raw pointer",
         "unsafe fn raw_pointer_demo() {\n    let x = 42i32;\n    let r = &x as *const i32;\n    "),
        ("unsafe send sync",
         "struct MyWrapper(*mut u8);\n\nunsafe impl Send for MyWrapper {}\nunsafe impl Sync for MyWrapper {}\n\nimpl MyWrapper {\n    fn new(ptr: *mut u8) -> Self {\n        "),
        ("transmute safety comment",
         "// Safety: T and U must have the same size and alignment\nunsafe fn reinterpret<T, U>(val: T) -> U {\n    "),
    ],
    "misc": [
        ("derive default",
         "#[derive(Debug, Clone, PartialEq, Eq, Hash)]\nstruct Config {\n    host: String,\n    port: u16,\n    max_connections: usize,\n}\n\nimpl Default for Config {\n    fn default() -> Self {\n        "),
        ("enum match",
         "#[derive(Debug)]\nenum Shape {\n    Circle(f64),\n    Rectangle(f64, f64),\n    Triangle(f64, f64, f64),\n}\n\nimpl Shape {\n    fn area(&self) -> f64 {\n        match self {\n            "),
        ("builder pattern",
         "#[derive(Debug, Default)]\nstruct QueryBuilder {\n    table: String,\n    conditions: Vec<String>,\n    limit: Option<usize>,\n}\n\nimpl QueryBuilder {\n    pub fn new(table: &str) -> Self {\n        "),
        ("generic fn",
         "fn largest<T: PartialOrd>(list: &[T]) -> &T {\n    let mut largest = &list[0];\n    for item in list {\n        "),
    ],
}


# ---------------------------------------------------------------------------
# Quality heuristics
# ---------------------------------------------------------------------------

def score_output(prompt: str, completion: str) -> dict:
    full  = prompt + completion
    words = completion.split()

    brace_balance = full.count("{") - full.count("}")
    repeats   = sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])
    rep_ratio = repeats / max(len(words), 1)
    unique_ratio = len(set(words)) / max(len(words), 1)

    rust_keywords = {
        "fn", "let", "mut", "pub", "impl", "struct", "enum", "trait",
        "match", "if", "else", "return", "use", "mod", "async", "await",
        "Result", "Option", "Vec", "String", "self", "Self", "where",
    }
    keyword_hits = len(rust_keywords & set(words))
    clean_end    = completion.rstrip().endswith(("}", ";", '"', ")", ">"))

    return {
        "brace_balance": brace_balance,
        "rep_ratio":     round(rep_ratio, 3),
        "unique_ratio":  round(unique_ratio, 3),
        "keyword_hits":  keyword_hits,
        "clean_end":     clean_end,
        "tokens_out":    len(words),
    }


def format_score(s: dict) -> str:
    flags = []
    if s["brace_balance"] != 0:
        flags.append(f"braces {s['brace_balance']:+d}")
    if s["rep_ratio"] > 0.1:
        flags.append(f"repetitive {s['rep_ratio']:.0%}")
    if s["unique_ratio"] < 0.3:
        flags.append(f"low diversity {s['unique_ratio']:.0%}")
    if s["keyword_hits"] < 2:
        flags.append("few keywords")
    if not s["clean_end"]:
        flags.append("truncated?")
    quality = "✓ ok" if not flags else "⚠  " + " | ".join(flags)
    return f"[{quality}]  {s['tokens_out']} tokens"


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate(
    model:       RustSLM,
    tokenizer:   Tokenizer,
    prompt:      str,
    max_new:     int,
    temperature: float,
    top_k:       int,
    rep_penalty: float,
    rep_window:  int,
    device:      torch.device,
) -> tuple[str, float]:
    """Returns (completion_text, tokens_per_second)."""
    enc    = tokenizer.encode(prompt)
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    ids    = [i for i in enc.ids if i not in (bos_id, eos_id)]

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    kv_caches = [{} for _ in range(model.cfg.n_layers)]
    all_ids   = list(ids)

    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new):
            ctx       = input_ids if not kv_caches[0] else input_ids[:, -1:]
            logits, _ = model(ctx, kv_caches=kv_caches)
            logits    = logits[:, -1, :] / max(temperature, 1e-8)

            if rep_penalty > 1.0:
                for token_id in set(all_ids[-rep_window:]):
                    logits[0, token_id] /= rep_penalty

            if top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, -1:]] = float("-inf")

            next_id   = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            next_id_i = next_id.item()
            input_ids = next_id
            all_ids.append(next_id_i)

            if eos_id is not None and next_id_i == eos_id:
                break

    elapsed   = time.perf_counter() - t0
    new_ids   = all_ids[len(ids):]
    tok_per_s = len(new_ids) / max(elapsed, 1e-6)
    new_text  = tokenizer.decode(new_ids, skip_special_tokens=False)
    return new_text, tok_per_s


# ---------------------------------------------------------------------------
# Test suite runner
# ---------------------------------------------------------------------------

def run_test_suite(
    model, tokenizer, device,
    categories:      list[str],
    settings:        dict,
    save_report:     bool = False,
    checkpoint_path: str  = "",
    log_dir:         Path = Path(DEFAULT_LOG_DIR),
) -> list[dict]:
    results = []

    for cat in categories:
        if cat not in TEST_SUITE:
            print(f"  Unknown category: {cat}  (valid: {', '.join(TEST_SUITE)})")
            continue

        print(f"\n{'═' * 60}")
        print(f"  Category: {cat.upper()}  ({len(TEST_SUITE[cat])} prompts)")
        print(f"{'═' * 60}")

        for label, prompt in TEST_SUITE[cat]:
            print(f"\n── {label} ──")
            print(prompt, end="", flush=True)

            completion, tok_s = generate(
                model, tokenizer, prompt, device=device, **settings
            )
            print(completion)
            score = score_output(prompt, completion)
            print(f"  {format_score(score)}  ({tok_s:.0f} tok/s)")

            results.append({
                "category":   cat,
                "label":      label,
                "prompt":     prompt,
                "completion": completion,
                "score":      score,
                "tok_per_s":  round(tok_s, 1),
            })

    # summary
    print(f"\n{'═' * 60}")
    print(f"  TEST SUMMARY  —  {len(results)} prompts  |  {len(categories)} categories")
    print(f"{'═' * 60}")
    ok      = sum(1 for r in results
                  if r["score"]["rep_ratio"] < 0.1 and r["score"]["keyword_hits"] >= 2)
    rep_bad = sum(1 for r in results if r["score"]["rep_ratio"] >= 0.1)
    kw_bad  = sum(1 for r in results if r["score"]["keyword_hits"] < 2)
    avg_tok = sum(r["score"]["tokens_out"] for r in results) / max(len(results), 1)
    avg_spd = sum(r["tok_per_s"] for r in results) / max(len(results), 1)
    print(f"  Passed heuristics : {ok}/{len(results)}")
    print(f"  Repetitive        : {rep_bad}")
    print(f"  Low keyword hits  : {kw_bad}")
    print(f"  Avg tokens out    : {avg_tok:.0f}")
    print(f"  Avg tok/s         : {avg_spd:.0f}")

    if save_report:
        log_dir.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = log_dir / f"bench_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp":  ts,
                "checkpoint": checkpoint_path,
                "settings":   settings,
                "summary": {
                    "total": len(results), "passed": ok,
                    "repetitive": rep_bad, "low_keywords": kw_bad,
                    "avg_tokens": round(avg_tok), "avg_tok_per_s": round(avg_spd),
                },
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n  Report saved → {path}")

    return results


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device) -> tuple[RustSLM, ModelConfig]:
    print(f"Loading checkpoint : {checkpoint_path}")
    ckpt     = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("model_config", {})
    cfg      = ModelConfig(**{k: v for k, v in cfg_dict.items()
                               if hasattr(ModelConfig, k)}) if cfg_dict else ModelConfig()
    model    = RustSLM(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    step     = ckpt.get("step", "?")
    val_loss = ckpt.get("best_val_loss", "?")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Step       : {step:,}" if isinstance(step, int) else f"  Step       : {step}")
    print(f"  Val loss   : {val_loss:.4f}" if isinstance(val_loss, float) else f"  Val loss   : {val_loss}")
    print(f"  Parameters : {n_params / 1e6:.1f}M")
    print(f"  Device     : {device}")
    return model, cfg


def load_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    tok_file = tokenizer_dir / "tokenizer.json"
    if not tok_file.exists():
        raise FileNotFoundError(
            f"tokenizer.json not found in {tokenizer_dir}\n"
            f"Run train_tokenizer.py first, or pass --tokenizer to the right path."
        )
    tok = Tokenizer.from_file(str(tok_file))
    print(f"Tokenizer  : {tok_file}  (vocab={tok.get_vocab_size():,})")
    return tok


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

HELP_TEXT = """
─────────────────────────────────────────────────────────────
  COMMANDS
─────────────────────────────────────────────────────────────
  <prompt>           type a Rust prefix, hit Enter to complete
  \\n                use \\n for newlines in single-line input

  Sampling:
    :sample <n>      generate N completions for the NEXT prompt
                     great for checking output diversity

  Test suite:
    :test            run full automated test suite (all categories)
    :test <cat>      run one category:
                       ownership / lifetimes / async / traits /
                       errors / collections / closures / unsafe / misc
    :bench           run full suite + save timestamped JSON report

  Tuning:
    :temp  <float>   temperature     0.4=focused  0.7=balanced  1.0=creative
    :topk  <int>     top-k           20=tight  40=default  100=loose
    :max   <int>     max new tokens  (default 256)
    :rep   <float>   repetition penalty  1.0=off  1.3=moderate  2.0=strong
    :win   <int>     rep penalty window  (recent N tokens penalised)

  Other:
    :settings        show current settings
    :cats            list all test categories with prompt counts
    :save            save last output to log file
    :quit            exit
─────────────────────────────────────────────────────────────
"""


def repl(model, tokenizer, device, args, checkpoint_path: str) -> None:
    settings = {
        "temperature": args.temperature,
        "top_k":       args.top_k,
        "max_new":     args.max_new,
        "rep_penalty": args.rep_penalty,
        "rep_window":  args.rep_window,
    }
    log_dir     = Path(args.log_dir)
    last_prompt = ""
    last_output = ""
    sample_n    = 1

    print("\n" + "=" * 60)
    print("  RustSLM — pretrained inference")
    print("  TEXT COMPLETION mode — not instruction following.")
    print("  Type :help for all commands.")
    print("=" * 60)

    while True:
        try:
            raw = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw:
            continue

        if raw in (":quit", ":q", ":exit"):
            print("Bye!")
            break

        if raw == ":help":
            print(HELP_TEXT)
            continue

        if raw == ":settings":
            for k, v in settings.items():
                print(f"  {k:<14} : {v}")
            print(f"  {'sample_n':<14} : {sample_n}")
            print(f"  {'log_dir':<14} : {log_dir}")
            continue

        if raw == ":cats":
            for cat, prompts in TEST_SUITE.items():
                print(f"  {cat:<14} — {len(prompts)} prompts")
            continue

        if raw == ":save":
            if not last_output:
                print("  Nothing to save yet.")
                continue
            log_dir.mkdir(parents=True, exist_ok=True)
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = log_dir / f"output_{ts}.txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"Checkpoint : {checkpoint_path}\n")
                f.write(f"Settings   : {json.dumps(settings)}\n")
                f.write(f"Timestamp  : {ts}\n")
                f.write(f"\nPrompt:\n{last_prompt}\n\nCompletion:\n{last_output}\n")
            print(f"  Saved → {path}")
            continue

        if raw.startswith(":sample "):
            try:
                sample_n = int(raw.split()[1])
                print(f"  Next prompt will generate {sample_n} completions.")
            except (ValueError, IndexError):
                print("  Usage: :sample 3")
            continue

        if raw in (":test", ":bench") or raw.startswith(":test "):
            save  = raw == ":bench"
            parts = raw.split()
            cats  = [parts[1]] if len(parts) > 1 else list(TEST_SUITE.keys())
            run_test_suite(
                model, tokenizer, device,
                categories      = cats,
                settings        = settings,
                save_report     = save,
                checkpoint_path = checkpoint_path,
                log_dir         = log_dir,
            )
            continue

        # tuning knobs
        def _set(key, typ, usage):
            nonlocal settings
            try:
                settings[key] = typ(raw.split()[1])
                print(f"  {key} → {settings[key]}")
            except (ValueError, IndexError):
                print(f"  Usage: {usage}")

        if raw.startswith(":temp "):
            _set("temperature", float, ":temp 0.7"); continue
        if raw.startswith(":topk "):
            _set("top_k", int, ":topk 40"); continue
        if raw.startswith(":max "):
            _set("max_new", int, ":max 256"); continue
        if raw.startswith(":rep "):
            _set("rep_penalty", float, ":rep 1.3"); continue
        if raw.startswith(":win "):
            _set("rep_window", int, ":win 64"); continue

        if raw.startswith(":"):
            print(f"  Unknown command: {raw}  (type :help)")
            continue

        # generation
        prompt      = raw.replace("\\n", "\n")
        last_prompt = prompt

        for i in range(sample_n):
            header = f"[{i+1}/{sample_n}]  " if sample_n > 1 else ""
            print(f"\n{'─' * 60}  {header}")
            print(prompt, end="", flush=True)
            try:
                completion, tok_s = generate(
                    model, tokenizer, prompt, device=device, **settings
                )
                print(completion)
                score = score_output(prompt, completion)
                print(f"  {format_score(score)}  ({tok_s:.0f} tok/s)")
                if i == sample_n - 1:
                    last_output = completion
            except Exception as e:
                print(f"\n  [ERROR] {e}")

        print(f"{'─' * 60}")
        sample_n = 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive inference + test suite for RustSLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",   type=str,   default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer",    type=str,   default=DEFAULT_TOKENIZER)
    parser.add_argument("--temperature",  type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_k",        type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--max_new",      type=int,   default=DEFAULT_MAX_NEW)
    parser.add_argument("--rep_penalty",  type=float, default=DEFAULT_REP_PENALTY)
    parser.add_argument("--rep_window",   type=int,   default=DEFAULT_REP_WINDOW)
    parser.add_argument("--log_dir",      type=str,   default=DEFAULT_LOG_DIR)
    parser.add_argument("--cpu",          action="store_true",
                        help="Force CPU inference")
    parser.add_argument("--test",         action="store_true",
                        help="Run full automated test suite and exit")
    parser.add_argument("--bench",        action="store_true",
                        help="Run full test suite, save JSON report, and exit")
    parser.add_argument("--category",     type=str,   default=None,
                        help="Limit --test/--bench to one category")
    args = parser.parse_args()

    device    = torch.device("cpu") if args.cpu else \
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _  = load_model(Path(args.checkpoint), device)
    tokenizer = load_tokenizer(Path(args.tokenizer))

    settings = {
        "temperature": args.temperature,
        "top_k":       args.top_k,
        "max_new":     args.max_new,
        "rep_penalty": args.rep_penalty,
        "rep_window":  args.rep_window,
    }

    if args.test or args.bench:
        cats = [args.category] if args.category else list(TEST_SUITE.keys())
        run_test_suite(
            model, tokenizer, device,
            categories      = cats,
            settings        = settings,
            save_report     = args.bench,
            checkpoint_path = args.checkpoint,
            log_dir         = Path(args.log_dir),
        )
    else:
        repl(model, tokenizer, device, args, checkpoint_path=args.checkpoint)