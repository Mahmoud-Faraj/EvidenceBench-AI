"""Download and prepare the licensed SciFact development benchmark."""

from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "external" / "scifact.tar.gz"
RAW_DIR = ROOT / "data" / "external" / "scifact"
PROCESSED_DIR = ROOT / "data" / "processed" / "scifact"


def verify_checksum(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"Checksum mismatch: expected {SHA256}, received {digest}")


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if not target.is_relative_to(root):
            raise RuntimeError(f"Unsafe archive member: {member.name}")
    archive.extractall(destination, filter="data")


def locate(filename: str) -> Path:
    matches = list(RAW_DIR.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {filename}, found {len(matches)}")
    return matches[0]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare() -> None:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, ARCHIVE)
    verify_checksum(ARCHIVE)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        safe_extract(archive, RAW_DIR)

    corpus_rows = read_jsonl(locate("corpus.jsonl"))
    claim_rows = read_jsonl(locate("claims_dev.jsonl"))
    documents = [
        {
            "doc_id": str(row["doc_id"]),
            "title": row["title"],
            "text": " ".join(row["abstract"]),
            "language": "en",
            "metadata": {"source": "SciFact"},
        }
        for row in corpus_rows
    ]
    examples = []
    for row in claim_rows:
        relevant = tuple(str(doc_id) for doc_id in row.get("evidence", {}))
        if relevant:
            examples.append(
                {
                    "question_id": str(row["id"]),
                    "question": row["claim"],
                    "relevant_doc_ids": relevant,
                    "language": "en",
                }
            )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "documents.json").write_text(
        json.dumps(documents, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (PROCESSED_DIR / "evaluation.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Prepared {len(documents):,} documents and {len(examples):,} labeled claims")


if __name__ == "__main__":
    prepare()
