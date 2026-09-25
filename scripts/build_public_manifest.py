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
        path for path in _files() if path != OUTPUT
    )
    lines = [f"{_sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(lines)} hashes to {OUTPUT.name}")


def _files() -> list[Path]:
    files: list[Path] = []
    for directory, subdirectories, filenames in ROOT.walk():
        subdirectories[:] = [
            name for name in subdirectories
            if name not in {".git", "__pycache__"}
            and not (directory == ROOT and name == "private")
        ]
        files.extend(directory / name for name in filenames)
    return files


if __name__ == "__main__":
    main()
