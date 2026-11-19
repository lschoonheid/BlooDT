"""Local cache of the VitalDB dataset (PhysioNet, ~95 GB).

Files are fetched on first use and kept under CACHE_DIR in the same layout as
https://physionet.org/files/vitaldb/1.0.0/, so a cache filled by `download()`
(or by a plain mirror of PhysioNet) is used as-is. Every download is checked
against the dataset's SHA256SUMS.txt.

    from digital_twin_engineering import vitaldb_cache as vdb

    clinical = vdb.table("clinical_data")
    hr = vdb.load_case(1, ["Solar8000/HR"], interval=1)

Pre-download from the shell: `uv run vitaldb-cache download --help`.
"""

import argparse
import hashlib
import os
import sys
import threading
import time
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import vitaldb

BASE_URL = "https://physionet.org/files/vitaldb/1.0.0/"
CACHE_DIR = Path(
    os.environ.get("VITALDB_CACHE_DIR")
    or Path(__file__).resolve().parents[2] / "data" / "vitaldb"
)
TABLES = ("clinical_data", "clinical_parameters", "lab_data", "lab_parameters", "track_names")
_CHUNK = 1 << 20
_RETRIES = 3


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _download(rel: str, sha256: str | None) -> Path:
    """Download `rel` into the cache atomically (via a .part file), checking its hash."""
    dest = CACHE_DIR / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Unique per thread, so concurrent fetches of the same file can't interleave.
    part = dest.with_name(f"{dest.name}.{os.getpid()}-{threading.get_ident()}.part")
    for attempt in range(1, _RETRIES + 1):
        try:
            h = hashlib.sha256()
            with urllib.request.urlopen(BASE_URL + rel, timeout=60) as resp, part.open("wb") as f:
                while chunk := resp.read(_CHUNK):
                    h.update(chunk)
                    f.write(chunk)
            if sha256 is not None and h.hexdigest() != sha256:
                raise OSError(f"checksum mismatch for {rel}")
            part.replace(dest)
            return dest
        except OSError:
            part.unlink(missing_ok=True)
            if attempt == _RETRIES:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


@cache
def manifest() -> dict[str, str]:
    """Map of every file in the dataset (path relative to CACHE_DIR) to its SHA-256."""
    path = CACHE_DIR / "SHA256SUMS.txt"
    if not path.exists():
        _download("SHA256SUMS.txt", None)
    sums = {}
    for line in path.read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        sums[rel] = digest
    return sums


def fetch(rel: str) -> Path:
    """Local path of a dataset file, downloading it first if it isn't cached."""
    path = CACHE_DIR / rel
    if path.exists():
        return path
    return _download(rel, manifest()[rel])


def case_ids() -> list[int]:
    """All case IDs in the dataset."""
    return sorted(
        int(Path(rel).stem) for rel in manifest() if rel.startswith("vital_files/")
    )


def case_path(caseid: int) -> Path:
    """Local path of a case's .vital file, downloading it first if needed."""
    return fetch(f"vital_files/{caseid:04d}.vital")


def table(name: str) -> pd.DataFrame:
    """One of the dataset's CSV tables, e.g. "clinical_data" (see TABLES)."""
    if name not in TABLES:
        raise ValueError(f"unknown table {name!r}, expected one of {TABLES}")
    return pd.read_csv(fetch(f"{name}.csv"))


def load_vital(caseid: int, tracks: list[str] | None = None) -> vitaldb.VitalFile:
    """A case as a vitaldb.VitalFile, optionally restricted to `tracks`."""
    return vitaldb.VitalFile(str(case_path(caseid)), tracks)


def load_case(caseid: int, tracks: list[str], interval: float = 1) -> np.ndarray:
    """Tracks of a case resampled every `interval` seconds, one column per track.

    Same shape as vitaldb.load_case, but read from the local cache instead of the
    VitalDB web API.
    """
    return load_vital(caseid, tracks).to_numpy(tracks, interval)


def download(
    caseids: Iterable[int] | None = None, jobs: int = 8, verify: bool = True
) -> list[str]:
    """Pre-download the CSV tables and the given cases (default: all of them).

    Files already in the cache are skipped; with `verify` they are hashed first and
    re-downloaded if they don't match, which also catches interrupted downloads
    from other tools. Returns the files that failed.
    """
    sums = manifest()
    if caseids is None:
        rels = list(sums)
    else:
        rels = [rel for rel in sums if not rel.startswith("vital_files/")]
        rels += [f"vital_files/{caseid:04d}.vital" for caseid in caseids]

    def ensure(rel: str) -> str:
        path = CACHE_DIR / rel
        if path.exists() and (not verify or _sha256(path) == sums[rel]):
            return "cached"
        _download(rel, sums[rel])
        return "downloaded"

    failed = []
    with ThreadPoolExecutor(jobs) as pool:
        futures = {pool.submit(ensure, rel): rel for rel in rels}
        for i, future in enumerate(as_completed(futures), 1):
            rel = futures[future]
            try:
                status = future.result()
            except Exception as e:
                failed.append(rel)
                status = f"FAILED ({e})"
            print(f"[{i}/{len(rels)}] {rel}: {status}", flush=True)
    return failed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vitaldb-cache", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="pre-download cases into the cache")
    dl.add_argument("caseids", nargs="*", type=int, help="cases to fetch (default: all)")
    dl.add_argument("-j", "--jobs", type=int, default=8, help="parallel downloads (default: 8)")
    dl.add_argument(
        "--no-verify", action="store_true", help="trust cached files without hashing them"
    )
    args = parser.parse_args(argv)

    print(f"Cache: {CACHE_DIR}", flush=True)
    failed = download(args.caseids or None, args.jobs, not args.no_verify)
    if failed:
        print(f"{len(failed)} file(s) failed: {' '.join(failed)}", file=sys.stderr)
        sys.exit(1)
    print("Done.")
