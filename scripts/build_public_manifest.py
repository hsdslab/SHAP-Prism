#!/usr/bin/env python3
"""Write a deterministic SHA-256 manifest for the public-source payload."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "PUBLIC_SOURCE_SHA256SUMS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != OUTPUT
        and ".git" not in path.relative_to(ROOT).parts
        and "__pycache__" not in path.relative_to(ROOT).parts
    )
    lines = [f"{_sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} hashes to {OUTPUT.name}")


if __name__ == "__main__":
    main()
