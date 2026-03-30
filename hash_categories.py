"""
rehash_categories.py
====================
Replaces integer `id` fields in rust_categories.jsonl with deterministic
SHA-256 hashes derived from the category name.

Why deterministic?
  The hash is computed from the category string itself, so the same category
  always gets the same ID — regardless of file order, insertion, or reruns.
  This is what makes the versioning system work: every conversation, lecture,
  and example generated for a category can reference its hash ID, and that
  reference stays valid forever even as the dataset grows or is reordered.

Hash format:
  First 12 hex characters of SHA-256("category name")
  e.g. "Core - Move Semantics on Function Call" → "a3f9c21b8e04"

  12 hex chars = 48 bits of space = 281 trillion possible values.
  Collision probability across 1M categories is ~1 in 10^9. Safe.

Usage:
    python rehash_categories.py
    python rehash_categories.py --input rust_categories.jsonl --output rust_categories_hashed.jsonl
    python rehash_categories.py --verify   # check for collisions only, no output written

Output:
    rust_categories_hashed.jsonl  — same records, id field replaced with hash string
    A collision report is printed if any two categories produce the same hash (should never happen).
"""

import argparse
import hashlib
import json
from pathlib import Path


HASH_LENGTH = 12  # hex characters — 48 bits, plenty for any realistic category count


def category_hash(category_name: str) -> str:
    """Deterministic 12-char hex hash derived from the category name."""
    digest = hashlib.sha256(category_name.encode("utf-8")).hexdigest()
    return digest[:HASH_LENGTH]


def load_records(path: Path) -> list[dict]:
    records = []
    errors  = 0
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Line {line_num}: JSON parse error — {e}")
                errors += 1
    if errors:
        print(f"  {errors} malformed lines skipped.")
    return records


def check_collisions(records: list[dict]) -> dict[str, list[str]]:
    """Return a dict of hash → [category names] for any hash that appears more than once."""
    seen: dict[str, list[str]] = {}
    for rec in records:
        cat  = rec["category"]
        h    = category_hash(cat)
        seen.setdefault(h, []).append(cat)
    return {h: cats for h, cats in seen.items() if len(cats) > 1}


def rehash(input_path: Path, output_path: Path, verify_only: bool = False) -> None:
    print(f"Loading  : {input_path}")
    records = load_records(input_path)
    print(f"Records  : {len(records):,}")

    # collision check
    collisions = check_collisions(records)
    if collisions:
        print(f"\n!! COLLISION WARNING — {len(collisions)} hash collision(s) detected:")
        for h, cats in collisions.items():
            print(f"   {h}: {cats}")
        print("   Increase HASH_LENGTH or disambiguate category names before proceeding.")
        return
    else:
        print(f"Collisions: none  (all {len(records):,} hashes are unique)")

    if verify_only:
        print("--verify mode: no output written.")
        return

    # replace ids
    updated = []
    for rec in records:
        new_rec      = dict(rec)
        new_rec["id"] = category_hash(rec["category"])
        updated.append(new_rec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in updated:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Written  : {output_path}  ({len(updated):,} records)")
    print()
    print("Sample hashes:")
    for rec in updated[:5]:
        print(f"  {rec['id']}  ←  {rec['category']}")
    print()
    print("Verify a hash manually:")
    print("  python -c \"import hashlib; print(hashlib.sha256('Core - Move Semantics on Function Call'.encode()).hexdigest()[:12])\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replace integer category IDs with deterministic SHA-256 hashes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="rust_categories.jsonl",
        help="Input JSONL file with integer IDs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rust_categories_hashed.jsonl",
        help="Output JSONL file with hash IDs",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check for collisions only — do not write output file",
    )
    args = parser.parse_args()

    rehash(
        input_path  = Path(args.input),
        output_path = Path(args.output),
        verify_only = args.verify,
    )