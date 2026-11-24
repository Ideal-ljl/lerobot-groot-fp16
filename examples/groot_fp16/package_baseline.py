"""Seal a completed baseline directory with checksums and a verified tar.gz archive."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    root = args.bundle
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Cannot package an incomplete baseline")
    archive = root.with_name(root.name + ".tar.gz")
    if archive.exists():
        raise FileExistsError(archive)
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    checksums = {str(path.relative_to(root)): sha256(path) for path in paths}
    (root / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in checksums.items()))
    checksums["SHA256SUMS"] = sha256(root / "SHA256SUMS")
    with tarfile.open(archive, "w:gz", compresslevel=1) as handle:
        handle.add(root, arcname=root.name)
    checked = set()
    with tarfile.open(archive, "r|gz") as handle:
        for member in handle:
            if not member.isfile():
                continue
            name = str(Path(member.name).relative_to(root.name))
            digest = hashlib.sha256()
            with handle.extractfile(member) as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != checksums[name]:
                raise ValueError(f"Archive checksum mismatch: {name}")
            checked.add(name)
    if checked != set(checksums):
        raise ValueError("Archive is missing files")
    digest = sha256(archive)
    archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n")
    print(
        json.dumps(
            {
                "archive": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": digest,
                "files_verified": len(checked),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
