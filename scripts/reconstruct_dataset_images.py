import argparse
import csv
import json
import mimetypes
import shutil
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct dataset/images/train and dataset/images/val from frozen metadata and captions."
    )
    parser.add_argument("--final-kept", default="dataset/metadata/final_kept_dataset.csv")
    parser.add_argument("--train-jsonl", default="dataset/captions/train.jsonl")
    parser.add_argument("--val-jsonl", default="dataset/captions/val.jsonl")
    parser.add_argument("--raw-cache", default="rawData/final_shortlist_images")
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def guess_extension(url: str, content_type: str) -> str:
    path_suffix = Path(urlparse(url).path).suffix.lower()
    if path_suffix in VALID_EXTENSIONS:
        return path_suffix
    guessed = mimetypes.guess_extension((content_type or "").split(";")[0].strip().lower())
    if guessed in VALID_EXTENSIONS:
        return guessed
    return ".jpg"


def download_to(url: str, dest: Path, timeout: int, session: requests.Session) -> None:
    try:
        head = session.head(url, allow_redirects=True, timeout=timeout)
        content_type = head.headers.get("Content-Type", "")
    except requests.RequestException:
        content_type = ""

    if not dest.suffix:
        dest = dest.with_suffix(guess_extension(url, content_type))

    with session.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def reconstruct_split(records: list[dict], kept_by_name: dict[str, dict[str, str]], raw_cache: Path, timeout: int) -> tuple[int, int]:
    session = requests.Session()
    session.headers.update({"User-Agent": "sdImplntn-reconstruct/1.0"})

    restored = 0
    missing = 0
    progress = tqdm(records, desc="Reconstructing split", unit="img")
    for record in progress:
        relative = record["file_name"]
        dest = Path(relative)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            restored += 1
            progress.set_postfix(restored=restored, missing=missing)
            continue

        basename = dest.name
        meta = kept_by_name.get(basename)
        if not meta:
            missing += 1
            progress.set_postfix(restored=restored, missing=missing)
            continue

        cached = raw_cache / basename
        if cached.exists() and cached.stat().st_size > 0:
            shutil.copy2(cached, dest)
            restored += 1
            progress.set_postfix(restored=restored, missing=missing)
            continue

        url = (meta.get("url") or "").strip()
        if url:
            try:
                download_to(url, dest, timeout, session)
                restored += 1
            except requests.RequestException:
                missing += 1
        else:
            # OpenBrush rows have no URL and require the original exported file/cache.
            missing += 1

        progress.set_postfix(restored=restored, missing=missing)

    progress.close()
    return restored, missing


def main() -> None:
    args = parse_args()
    final_kept = Path(args.final_kept).resolve()
    train_jsonl = Path(args.train_jsonl).resolve()
    val_jsonl = Path(args.val_jsonl).resolve()
    raw_cache = Path(args.raw_cache).resolve()

    kept_rows = read_csv(final_kept)
    kept_by_name = {row.get("file_name", ""): row for row in kept_rows}

    train_records = read_jsonl(train_jsonl)
    val_records = read_jsonl(val_jsonl)

    train_restored, train_missing = reconstruct_split(train_records, kept_by_name, raw_cache, args.timeout)
    val_restored, val_missing = reconstruct_split(val_records, kept_by_name, raw_cache, args.timeout)

    print(f"Train restored: {train_restored}, train missing: {train_missing}")
    print(f"Val restored: {val_restored}, val missing: {val_missing}")
    if train_missing or val_missing:
        print("Note: rows without a URL (for example OpenBrush) require the cached exported image file to still exist.")


if __name__ == "__main__":
    main()
