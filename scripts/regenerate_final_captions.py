import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate final train/val captions by joining kept rows back to source metadata.")
    parser.add_argument("--final-kept", default="dataset/metadata/final_kept_dataset.csv")
    parser.add_argument("--pd12m-source", default="rawData/metadata/domain_classified_candidates.csv")
    parser.add_argument("--relaion-source", default="rawData/relaion/metadata/short_domain_candidates.csv")
    parser.add_argument("--openbrush-source", default="rawData/openbrush/metadata/target_domain_candidates.csv")
    parser.add_argument("--train-jsonl", default="dataset/captions/train.jsonl")
    parser.add_argument("--val-jsonl", default="dataset/captions/val.jsonl")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def synthesize_openbrush_caption(row: dict[str, str]) -> str:
    subject = (row.get("subject") or "").strip().rstrip(".")
    setting = (row.get("setting") or "").strip().rstrip(".")
    lighting = (row.get("lighting") or "").strip().rstrip(".")
    mood = (row.get("mood") or "").strip().rstrip(".")

    parts = []
    if subject:
        parts.append(subject)
    if setting:
        parts.append(setting)
    if lighting:
        parts.append(lighting)
    elif mood:
        parts.append(mood)

    text = ". ".join(parts[:2]).strip()
    if not text:
        text = (row.get("caption_full") or "").strip()
    if not text:
        text = subject
    if text and not text.endswith("."):
        text += "."
    return text


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def rewrite_split(path: Path, kept_by_basename: dict[str, dict[str, str]], pd12m_by_url: dict[str, str], relaion_by_url: dict[str, str], openbrush_by_id: dict[str, str]) -> int:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    rewritten = 0
    for record in records:
        basename = Path(record["file_name"]).name
        kept = kept_by_basename.get(basename)
        if not kept:
            continue

        source = kept.get("dataset_source", "")
        caption = ""
        if source == "PD12M":
            caption = pd12m_by_url.get(kept.get("url", ""), "")
        elif source == "Re-LAION":
            caption = relaion_by_url.get(kept.get("url", ""), "")
        elif source == "OpenBrush":
            caption = openbrush_by_id.get(kept.get("id", ""), "")

        record["text"] = caption
        rewritten += 1

    write_jsonl(path, records)
    return rewritten


def main() -> None:
    args = parse_args()
    final_kept_path = Path(args.final_kept).resolve()
    pd12m_path = Path(args.pd12m_source).resolve()
    relaion_path = Path(args.relaion_source).resolve()
    openbrush_path = Path(args.openbrush_source).resolve()
    train_jsonl = Path(args.train_jsonl).resolve()
    val_jsonl = Path(args.val_jsonl).resolve()

    kept_rows = read_csv(final_kept_path)
    kept_by_basename = {row.get("file_name", ""): row for row in kept_rows}

    pd12m_rows = read_csv(pd12m_path)
    pd12m_by_url = {row.get("url", ""): (row.get("caption") or "").strip() for row in pd12m_rows}

    relaion_rows = read_csv(relaion_path)
    relaion_by_url = {row.get("url", ""): (row.get("caption") or "").strip() for row in relaion_rows}

    openbrush_rows = read_csv(openbrush_path)
    openbrush_by_id = {row.get("id", ""): synthesize_openbrush_caption(row) for row in openbrush_rows}

    train_rewritten = rewrite_split(train_jsonl, kept_by_basename, pd12m_by_url, relaion_by_url, openbrush_by_id)
    val_rewritten = rewrite_split(val_jsonl, kept_by_basename, pd12m_by_url, relaion_by_url, openbrush_by_id)

    print(f"Train captions rewritten: {train_rewritten}")
    print(f"Val captions rewritten: {val_rewritten}")
    print(f"Train JSONL: {train_jsonl}")
    print(f"Val JSONL: {val_jsonl}")


if __name__ == "__main__":
    main()
