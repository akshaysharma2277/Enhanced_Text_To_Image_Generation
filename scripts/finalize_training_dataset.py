import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


FINAL_TARGETS = {
    "Nature": 140,
    "Architecture": 140,
    "Sci-Fi": 140,
    "Fantasy": 140,
    "Historical": 140,
    "Abstract": 140,
    "Artistic": 140,
    "Wildlife": 140,
    "Everyday Life": 140,
    "Emotions": 135,
}

VAL_TARGETS = {
    "Nature": 14,
    "Architecture": 14,
    "Sci-Fi": 14,
    "Fantasy": 14,
    "Historical": 14,
    "Abstract": 14,
    "Artistic": 14,
    "Wildlife": 14,
    "Everyday Life": 14,
    "Emotions": 14,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize the training-ready dataset from accepted validated sets.")
    parser.add_argument(
        "--base-accepted",
        default="dataset/metadata/validated_images.csv",
        help="Base accepted validated CSV.",
    )
    parser.add_argument(
        "--extra-accepted",
        action="append",
        default=["dataset/metadata/openbrush_validated_increment.csv"],
        help="Additional accepted increment CSVs. Can be passed multiple times.",
    )
    parser.add_argument(
        "--final-metadata-output",
        default="dataset/metadata/final_kept_dataset.csv",
        help="Output CSV for the final kept dataset.",
    )
    parser.add_argument(
        "--summary-output",
        default="dataset/metadata/final_kept_summary.csv",
        help="Output CSV for final per-domain summary.",
    )
    parser.add_argument(
        "--train-jsonl",
        default="dataset/captions/train.jsonl",
        help="Output train JSONL path.",
    )
    parser.add_argument(
        "--val-jsonl",
        default="dataset/captions/val.jsonl",
        help="Output val JSONL path.",
    )
    parser.add_argument(
        "--train-dir",
        default="dataset/images/train",
        help="Output train image directory.",
    )
    parser.add_argument(
        "--val-dir",
        default="dataset/images/val",
        help="Output val image directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/val split.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def union_fieldnames(*field_sets: list[str]) -> list[str]:
    ordered: list[str] = []
    seen = set()
    for fields in field_sets:
        for field in fields:
            if field not in seen:
                ordered.append(field)
                seen.add(field)
    return ordered


def build_identity(row: dict[str, str]) -> tuple[str, str]:
    source = row.get("dataset_source", "")
    if source == "PD12M":
        return source, row.get("id", "") or row.get("hash", "") or row.get("url", "")
    if source == "Re-LAION":
        return source, row.get("sha256", "") or row.get("relative_path", "") or row.get("url", "")
    return source, row.get("id", "") or row.get("source_file", "") or row.get("file_name", "")


def row_sort_key(row: dict[str, str]) -> tuple[int, int, int]:
    strength = int(row.get("strength_score") or 0)
    domain_score = int(row.get("domain_score") or 0)
    blur_score = float(row.get("blur_score") or 0.0)
    contrast_score = float(row.get("contrast_score") or 0.0)
    return (-strength, -domain_score, -int(blur_score + contrast_score))


def synthesize_caption(row: dict[str, str]) -> str:
    caption = (row.get("text") or "").strip()
    if caption:
        return caption

    if row.get("dataset_source") == "OpenBrush":
        subject = (row.get("subject") or "").strip().rstrip(".")
        setting = (row.get("setting") or "").strip().rstrip(".")
        lighting = (row.get("lighting") or "").strip().rstrip(".")
        pieces = [piece for piece in [subject, setting, lighting] if piece]
        if pieces:
            text = ". ".join(pieces[:2])
            return text if text.endswith(".") else f"{text}."

    caption_full = (row.get("caption_full") or "").strip()
    if caption_full:
        parts = [line.strip() for line in caption_full.splitlines() if line.strip() and not line.startswith("**")]
        if parts:
            text = " ".join(parts[:2]).strip()
            return text if text.endswith(".") else f"{text}."

    subject = (row.get("subject") or "").strip()
    return subject if subject.endswith(".") else f"{subject}." if subject else ""


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    base_path = Path(args.base_accepted).resolve()
    extra_paths = [Path(path).resolve() for path in args.extra_accepted]
    final_metadata_output = Path(args.final_metadata_output).resolve()
    summary_output = Path(args.summary_output).resolve()
    train_jsonl = Path(args.train_jsonl).resolve()
    val_jsonl = Path(args.val_jsonl).resolve()
    train_dir = Path(args.train_dir).resolve()
    val_dir = Path(args.val_dir).resolve()

    final_metadata_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    train_jsonl.parent.mkdir(parents=True, exist_ok=True)
    val_jsonl.parent.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    rows, fields = read_csv(base_path)
    field_sets = [fields]

    for path in extra_paths:
        extra_rows, extra_fields = read_csv(path)
        rows.extend(extra_rows)
        field_sets.append(extra_fields)

    merged_fields = union_fieldnames(*field_sets)

    unique_rows = {}
    for row in rows:
        unique_rows[build_identity(row)] = row

    by_domain = defaultdict(list)
    for row in unique_rows.values():
        by_domain[row.get("domain", "")].append(row)

    for domain in by_domain:
        by_domain[domain].sort(key=row_sort_key)

    final_rows = []
    summary_rows = []

    for domain, target in FINAL_TARGETS.items():
        chosen = by_domain[domain][:target]
        final_rows.extend(chosen)
        summary_rows.append(
            {
                "domain": domain,
                "target": str(target),
                "selected": str(len(chosen)),
                "status": "OK" if len(chosen) == target else "SHORT",
            }
        )

    with final_metadata_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fields)
        writer.writeheader()
        writer.writerows(final_rows)

    with summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["domain", "target", "selected", "status"])
        writer.writeheader()
        writer.writerows(summary_rows)

    final_by_domain = defaultdict(list)
    for row in final_rows:
        final_by_domain[row.get("domain", "")].append(row)

    train_records = []
    val_records = []

    for domain, domain_rows in final_by_domain.items():
        shuffled = domain_rows[:]
        random.shuffle(shuffled)
        val_count = VAL_TARGETS[domain]
        val_rows = shuffled[:val_count]
        train_rows = shuffled[val_count:]

        for split_name, split_rows, split_dir, split_records in (
            ("val", val_rows, val_dir, val_records),
            ("train", train_rows, train_dir, train_records),
        ):
            for row in split_rows:
                src = Path(row.get("local_path", ""))
                file_name = row.get("file_name", "")
                if not src.exists() and file_name:
                    src = Path("rawData/final_shortlist_images").resolve() / file_name
                if not src.exists():
                    raise FileNotFoundError(f"Image not found for final export: {row}")

                dst = split_dir / file_name
                copy_image(src, dst)
                caption = synthesize_caption(row)
                split_records.append(
                    {
                        "file_name": f"images/{split_name}/{file_name}",
                        "text": caption,
                    }
                )

    with train_jsonl.open("w", encoding="utf-8") as handle:
        for record in train_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with val_jsonl.open("w", encoding="utf-8") as handle:
        for record in val_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Final kept rows: {len(final_rows)}")
    print(f"Train records: {len(train_records)}")
    print(f"Val records: {len(val_records)}")
    print(f"Final metadata: {final_metadata_output}")
    print(f"Summary: {summary_output}")
    print(f"Train JSONL: {train_jsonl}")
    print(f"Val JSONL: {val_jsonl}")


if __name__ == "__main__":
    main()
