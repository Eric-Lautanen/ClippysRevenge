# ClippysRevenge Dataset Schema — v1.0

A standardized record format for Rust AI training data.  Every record in
the dataset — regardless of source, contributor, or generation method —
conforms to this schema.  The goal is a single, well-structured corpus that
any researcher or hobbyist can load and use without preprocessing surprises.

---

## Full Schema

```json
{
  "schema_version":   "1.0",

  "type":             "bug_fix",

  "category":         "Async - Tokio spawn and join",
  "difficulty":       "intermediate",
  "concepts":         ["async", "tokio", "spawn", "join", "blocking"],
  "crates":           ["tokio"],
  "edition":          "2021",

  "prompt":           "I need to run two async tasks concurrently and collect both results...",

  "context_code":     "// optional: surrounding code that provides context but is not the focus",

  "broken_code":      "fn main() {\n    thread::sleep(Duration::from_secs(1));\n}",
  "error_message":    "thread::sleep blocks the async executor",
  "error_code":       "E0277",

  "code":             "fn main() {\n    tokio::time::sleep(Duration::from_secs(1)).await;\n}",

  "explanation":      "tokio::time::sleep yields the executor; thread::sleep blocks the whole thread",

  "has_unsafe":       false,
  "is_no_std":        false,

  "validation": {
    "fmt":            true,
    "clippy":         true,
    "build":          true,
    "test":           null,
    "clippy_lints":   []
  },

  "compiler_ver":     "rustc 1.86.0 (05f9846f8 2025-03-31)",
  "min_rust_version": "1.75.0",

  "source":           "human",
  "source_model":     null,
  "contributor":      "github:username",
  "license":          "Apache-2.0",
  "verified_at":      1741234567
}
```

---

## Field Reference

### Identity & Versioning

| Field            | Type   | Required | Description |
|------------------|--------|----------|-------------|
| `schema_version` | string | yes      | Schema version this record conforms to. Always `"1.0"` for this spec. Increment when breaking changes are made. |

---

### Classification

| Field        | Type            | Required | Description |
|--------------|-----------------|----------|-------------|
| `type`       | string (enum)   | yes      | The kind of example. See **Type Values** below. |
| `category`   | string          | yes      | Category from `rust_categories.jsonl`, e.g. `"Async - Tokio spawn and join"`. Free-form for community contributions. |
| `difficulty` | string (enum)   | yes      | `"beginner"` / `"intermediate"` / `"advanced"` |
| `concepts`   | string[]        | yes      | Flat tag array of Rust concepts covered. Used for filtered subset building. |
| `crates`     | string[]        | yes      | Crates required to compile this example. Empty array `[]` if stdlib only. |
| `edition`    | string (enum)   | yes      | Rust edition: `"2015"` / `"2018"` / `"2021"` / `"2024"` |

#### Type Values

| Value              | When to use |
|--------------------|-------------|
| `"bug_fix"`        | Before/after pair: broken code with an error, then the corrected version |
| `"code_example"`   | A single correct code demonstration of a concept or pattern |
| `"refactor"`       | Before/after pair: working but non-idiomatic code, then the improved version |
| `"api_usage"`      | Focused example of how to use a specific crate or stdlib API |
| `"explanation"`    | Conceptual explanation with illustrative code (may be short snippets) |
| `"qa_pair"`        | A question/answer exchange in conversational format |
| `"code_review"`    | A code snippet with inline review comments and suggested improvements |

---

### Content

| Field          | Type   | Required          | Description |
|----------------|--------|-------------------|-------------|
| `prompt`       | string | yes               | The question, task, or scenario that this example addresses. What a developer might type to a chatbot. |
| `context_code` | string | no (default null) | Supporting code that provides context (e.g. struct definitions, imports) but is not the primary focus. Helps the model understand the full picture. |
| `broken_code`  | string | if `type=bug_fix` or `type=refactor` | The before state — incorrect, non-idiomatic, or broken code. |
| `error_message`| string | if broken_code present | The compiler error, warning, clippy lint, or description of the problem. Use the actual `rustc` output where possible. |
| `error_code`   | string | no (default null) | The `rustc` error code if applicable (e.g. `"E0502"`). `null` for warnings or logical errors. |
| `code`         | string | yes               | The primary code output — the correct, idiomatic, validated answer. For `bug_fix` and `refactor` this is the fixed version. For all other types it is the example itself. |
| `explanation`  | string | yes               | A clear explanation of what the code does, why the approach is correct/idiomatic, and (for bug_fix/refactor) what was wrong and why the fix works. |

---

### Code Properties

| Field        | Type    | Required | Description |
|--------------|---------|----------|-------------|
| `has_unsafe` | boolean | yes      | `true` if the `code` field contains any `unsafe` block or `unsafe fn`. Allows consumers to control unsafe exposure in training. |
| `is_no_std`  | boolean | yes      | `true` if the example is written for a `#![no_std]` environment. |

---

### Validation

The `validation` object captures the result of each check run against the `code` field.

```json
"validation": {
  "fmt":         true,
  "clippy":      true,
  "build":       true,
  "test":        null,
  "clippy_lints": []
}
```

| Sub-field      | Type           | Description |
|----------------|----------------|-------------|
| `fmt`          | bool or null   | `true` = passes `cargo fmt --check`. `null` = not checked. |
| `clippy`       | bool or null   | `true` = passes `cargo clippy -- -D warnings`. `null` = not checked. |
| `build`        | bool or null   | `true` = `cargo build` succeeds at `min_rust_version`. `null` = not checked. |
| `test`         | bool or null   | `true` = all `cargo test` tests pass. `null` = no tests present or not run. |
| `clippy_lints` | string[]       | Lints that fired but were explicitly allowed (e.g. `["clippy::needless_return"]`). Empty array if none. |

**Rules:**
- A record is considered **fully validated** when `fmt`, `clippy`, and `build` are all `true`.
- A record with any `false` value **must not be included** in the clean dataset. Fix it or discard it.
- `null` means the check was skipped — acceptable for `test` if no tests exist, not acceptable for `build`.

---

### Provenance

| Field            | Type   | Required | Description |
|------------------|--------|----------|-------------|
| `compiler_ver`   | string | yes      | Full `rustc --version` output used during validation, e.g. `"rustc 1.86.0 (05f9846f8 2025-03-31)"` |
| `min_rust_version` | string | yes    | Minimum Rust version this compiles on (MSRV), e.g. `"1.75.0"`. Determined by MSRV checking. |
| `source`         | string (enum) | yes | Where this example originated. See **Source Values** below. |
| `source_model`   | string | if source=ai | The model that generated this example, e.g. `"claude-opus-4-5"`, `"gpt-4o-2024-11-20"`. `null` for human-authored. |
| `contributor`    | string | no       | GitHub username or handle of the contributor, e.g. `"github:username"`. `null` for automated pipelines. |
| `license`        | string | yes      | SPDX license identifier. All records in the core dataset use `"Apache-2.0"`. Community contributions must use a compatible license. |
| `verified_at`    | integer | yes     | Unix timestamp (seconds) when validation was last run. |

#### Source Values

| Value       | Meaning |
|-------------|---------|
| `"human"`   | Written entirely by a human contributor |
| `"ai"`      | Generated by an AI model (specify `source_model`) |
| `"ai_human"`| AI-generated then reviewed and edited by a human |
| `"extracted"` | Extracted from an existing dataset or source (specify in `contributor`) |

---

## Type-Specific Field Requirements

Different record types require different subsets of fields:

| Type           | `broken_code` | `error_message` | `error_code` | `context_code` |
|----------------|:---:|:---:|:---:|:---:|
| `bug_fix`      | required | required | recommended | optional |
| `code_example` | omit | omit | omit | optional |
| `refactor`     | required | required | omit | optional |
| `api_usage`    | omit | omit | omit | optional |
| `explanation`  | omit | omit | omit | optional |
| `qa_pair`      | omit | omit | omit | omit |
| `code_review`  | required | optional | omit | optional |

For omitted fields: **do not include the key at all**, or set to `null`. Do not include empty strings for fields that don't apply.

---

## Minimal Valid Records by Type

### bug_fix
```json
{
  "schema_version": "1.0",
  "type": "bug_fix",
  "category": "Async - Blocking inside async",
  "difficulty": "intermediate",
  "concepts": ["async", "tokio", "blocking"],
  "crates": ["tokio"],
  "edition": "2021",
  "prompt": "My async function hangs when I call thread::sleep inside it.",
  "broken_code": "async fn delay() { std::thread::sleep(Duration::from_secs(1)); }",
  "error_message": "thread::sleep blocks the async executor thread",
  "error_code": null,
  "code": "async fn delay() { tokio::time::sleep(Duration::from_secs(1)).await; }",
  "explanation": "std::thread::sleep blocks the OS thread, stalling the entire Tokio executor. tokio::time::sleep is async-aware and yields back to the runtime.",
  "has_unsafe": false,
  "is_no_std": false,
  "validation": { "fmt": true, "clippy": true, "build": true, "test": null, "clippy_lints": [] },
  "compiler_ver": "rustc 1.86.0 (05f9846f8 2025-03-31)",
  "min_rust_version": "1.75.0",
  "source": "ai",
  "source_model": "claude-opus-4-5",
  "contributor": null,
  "license": "Apache-2.0",
  "verified_at": 1741234567
}
```

### code_example
```json
{
  "schema_version": "1.0",
  "type": "code_example",
  "category": "Traits - Iterator adapter from scratch",
  "difficulty": "intermediate",
  "concepts": ["iterators", "traits", "generics"],
  "crates": [],
  "edition": "2021",
  "prompt": "Show me how to implement a custom iterator adapter that doubles every value.",
  "code": "struct Doubled<I> { inner: I }\nimpl<I: Iterator<Item=i32>> Iterator for Doubled<I> {\n    type Item = i32;\n    fn next(&mut self) -> Option<i32> { self.inner.next().map(|x| x * 2) }\n}",
  "explanation": "Doubled wraps any i32 iterator and multiplies each element by 2. The generic bound I: Iterator<Item=i32> ensures type safety without restricting to a concrete iterator type.",
  "has_unsafe": false,
  "is_no_std": false,
  "validation": { "fmt": true, "clippy": true, "build": true, "test": true, "clippy_lints": [] },
  "compiler_ver": "rustc 1.86.0 (05f9846f8 2025-03-31)",
  "min_rust_version": "1.56.0",
  "source": "ai_human",
  "source_model": "gpt-4o-2024-11-20",
  "contributor": "github:rustymind",
  "license": "Apache-2.0",
  "verified_at": 1741234567
}
```

---

## Changelog

| Version | Date       | Changes |
|---------|------------|---------|
| 1.0     | 2025-03-20 | Initial release |