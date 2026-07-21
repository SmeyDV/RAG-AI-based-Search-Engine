"""Export normalized movie metadata to JSON Lines.

Run from the repository root:
    python3 scripts/export_movies.py
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.ingest import load_documents  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "sample_docs")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "movies.jsonl")
    args = parser.parse_args()

    documents = load_documents(str(args.source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for document in documents:
            record = {
                "schema_version": 1,
                **document["metadata"],
                "raw_text": document["text"],
            }
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Exported {len(documents)} movies to {args.output}")


if __name__ == "__main__":
    main()
