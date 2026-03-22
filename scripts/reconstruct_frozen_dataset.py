import argparse
import csv
import io
import json
import mimetypes
import shutil
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq
import requests
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


PD12M_PREFIX = "https://pd12m.s3.us-west-2.amazonaws.com/"
RELAION_REPO_ID = "supermodelresearch/Re-LAION-Caption19M"
OPENBRUSH_REPO_ID = "jaddai/openbrush-75k"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct the frozen training-ready dataset from final metadata and split files."
    )
    parser.add_argument("--final-kept", default="dataset/metadata/final_kept_dataset.csv")
    parser.add_argument("--train-jsonl", default="dataset/captions/train.jsonl")
    parser.add_argument("--val-jsonl", default="dataset/captions/val.jsonl")
    parser.add_argument("--raw-cache", default="rawData/final_shortlist_images")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--download-openbrush-shards", action="store_true", help="Download OpenBrush parquet shards as needed.")
    parser.add_argument("--openbrush-cache-dir", default="rawData/openbrush_reconstruct_cache")
    parser.add_argument("--cleanup-openbrush-shards", action="store_true", help="Delete each downloaded OpenBrush shard after processing.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files already present in target folders.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def guess_extension(url: str, content_type: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in VALID_EXTENSIONS:
        return suffix
    guessed = mimetypes.guess_extension((content_type or "").split(";")[0].strip().lower())
    if guessed in VALID_EXTENSIONS:
        return guessed
    return ".jpg"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def reconstruct_http_file(url: str, dest: Path, timeout: int, session: requests.Session) -> None:
    try:
        head = session.head(url, allow_redirects=True, timeout=timeout)
        content_type = head.headers.get("Content-Type", "")
    except requests.RequestException:
        content_type = ""

    if not dest.suffix:
        dest = dest.with_suffix(guess_extension(url, content_type))

    ensure_parent(dest)
    with session.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def copy_from_cache(cache_path: Path, dest: Path) -> bool:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        ensure_parent(dest)
        shutil.copy2(cache_path, dest)
        return True
    return False


def export_openbrush_from_shards(
    target_rows: dict[str, dict[str, str]],
    output_targets: dict[str, Path],
    shard_cache_dir: Path,
    cleanup_openbrush_shards: bool,
) -> tuple[int, int]:
    shard_cache_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=OPENBRUSH_REPO_ID, repo_type="dataset")
    parquet_files = sorted(path for path in repo_files if path.startswith("data/") and path.endswith(".parquet"))

    exported = 0
    remaining = set(target_rows.keys())
    progress = tqdm(parquet_files, desc="OpenBrush reconstruct", unit="shard")
    for repo_path in progress:
        downloaded = Path(
            hf_hub_download(
                repo_id=OPENBRUSH_REPO_ID,
                repo_type="dataset",
                filename=repo_path,
                local_dir=str(shard_cache_dir),
                local_dir_use_symlinks=False,
            )
        )

        parquet = pq.ParquetFile(downloaded)
        for batch in parquet.iter_batches(columns=["id", "image"], batch_size=256):
            for row in batch.to_pylist():
                item_id = row.get("id", "")
                if item_id not in remaining:
                    continue
                image_struct = row.get("image") or {}
                image_bytes = image_struct.get("bytes")
                if not image_bytes:
                    continue
                out_path = output_targets[item_id]
                ensure_parent(out_path)
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image = image.convert("RGB")
                    image.save(out_path, format="PNG")
                remaining.remove(item_id)
                exported += 1
                if not remaining:
                    break
            if not remaining:
                break

        progress.set_postfix(exported=exported, remaining=len(remaining))
        if cleanup_openbrush_shards and downloaded.exists():
            downloaded.unlink()
            parent = downloaded.parent
            while parent != shard_cache_dir and parent.exists():
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
        if not remaining:
            break
    progress.close()
    return exported, len(remaining)


def build_split_targets(records: list[dict], kept_by_name: dict[str, dict[str, str]], split_root: str) -> list[tuple[dict[str, str], Path]]:
    result = []
    for record in records:
        basename = Path(record["file_name"]).name
        kept = kept_by_name.get(basename)
        if not kept:
            continue
        result.append((kept, Path(record["file_name"])))
    return result


def main() -> None:
    args = parse_args()
    final_kept_path = Path(args.final_kept).resolve()
    train_jsonl_path = Path(args.train_jsonl).resolve()
    val_jsonl_path = Path(args.val_jsonl).resolve()
    raw_cache = Path(args.raw_cache).resolve()
    openbrush_cache_dir = Path(args.openbrush_cache_dir).resolve()

    kept_rows = read_csv(final_kept_path)
    kept_by_name = {row.get("file_name", ""): row for row in kept_rows}
    train_records = read_jsonl(train_jsonl_path)
    val_records = read_jsonl(val_jsonl_path)

    split_targets = build_split_targets(train_records, kept_by_name, "train") + build_split_targets(val_records, kept_by_name, "val")
    session = requests.Session()
    session.headers.update({"User-Agent": "sdImplntn-reconstruct/1.0"})

    openbrush_targets: dict[str, dict[str, str]] = {}
    openbrush_output_paths: dict[str, Path] = {}

    restored = 0
    missing = 0
    progress = tqdm(split_targets, desc="Reconstruct frozen dataset", unit="img")
    for row, dest in progress:
        abs_dest = dest.resolve()
        if args.skip_existing and abs_dest.exists() and abs_dest.stat().st_size > 0:
            restored += 1
            progress.set_postfix(restored=restored, missing=missing)
            continue

        basename = row.get("file_name", "")
        cache_path = raw_cache / basename
        if copy_from_cache(cache_path, abs_dest):
            restored += 1
            progress.set_postfix(restored=restored, missing=missing)
            continue

        source = row.get("dataset_source", "")
        url = (row.get("url") or "").strip()

        if source in {"PD12M", "Re-LAION"} and url:
            try:
                reconstruct_http_file(url, abs_dest, args.timeout, session)
                restored += 1
            except requests.RequestException:
                missing += 1
        elif source == "OpenBrush":
            item_id = row.get("id", "")
            if item_id:
                openbrush_targets[item_id] = row
                openbrush_output_paths[item_id] = abs_dest
            else:
                missing += 1
        else:
            missing += 1

        progress.set_postfix(restored=restored, missing=missing, openbrush=len(openbrush_targets))
    progress.close()

    if openbrush_targets:
        if not args.download_openbrush_shards:
            print(f"OpenBrush pending: {len(openbrush_targets)}. Re-run with --download-openbrush-shards to reconstruct them.")
            missing += len(openbrush_targets)
        else:
            exported, remaining = export_openbrush_from_shards(
                openbrush_targets,
                openbrush_output_paths,
                openbrush_cache_dir,
                args.cleanup_openbrush_shards,
            )
            restored += exported
            missing += remaining

    print(f"Restored images: {restored}")
    print(f"Missing images: {missing}")
    print("Note: PD12M/Re-LAION are reconstructed from frozen URLs; OpenBrush requires --download-openbrush-shards.")


if __name__ == "__main__":
    main()
