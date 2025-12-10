"""
Convert W&I corpus 2024 to JSONL format for training.

Usage:
    uv run python prepare_data.py --input-dir ./corpus/whole-corpus --output-dir ./data

The 2024 corpus comes as a single TSV file with columns including:
- text: The essay text
- automarker_cefr_level: CEFR level from auto-marker (B1, B2, etc.)
- humannotator_cefr_level: CEFR level from human annotator (may be NA)
- split: train/dev/test (already pre-split!)
- is_final_version: TRUE/FALSE

We use the official splits and prefer human annotations when available.
"""
import csv
import json
from pathlib import Path
from collections import Counter

# CEFR level to numeric score (1-6 scale)
CEFR_TO_SCORE = {
    "A1": 1.0,
    "A2": 2.0,
    "B1": 3.0,
    "B2": 4.0,
    "C1": 5.0,
    "C2": 6.0,
}


def parse_wi_corpus_2024(input_dir: Path) -> dict[str, list[dict]]:
    """
    Parse the Write & Improve 2024 corpus.
    
    Returns:
        Dict with keys 'train', 'dev', 'test' containing essay lists.
    """
    # Find the main corpus file
    corpus_file = input_dir / "en-writeandimprove2024-corpus.tsv"
    
    if not corpus_file.exists():
        # Try to find any TSV file
        tsv_files = list(input_dir.glob("*.tsv"))
        if tsv_files:
            corpus_file = tsv_files[0]
        else:
            raise FileNotFoundError(f"No TSV file found in {input_dir}")
    
    print(f"Reading corpus from: {corpus_file}")
    
    essays = {"train": [], "dev": [], "test": []}
    skipped = {"no_cefr": 0, "not_final": 0, "invalid_split": 0}
    
    with open(corpus_file, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quotechar='"')
        
        for row in reader:
            # Only use final versions of essays
            is_final = row.get("is_final_version", "").upper() == "TRUE"
            if not is_final:
                skipped["not_final"] += 1
                continue
            
            # Get CEFR level (prefer human annotation, fallback to auto-marker)
            cefr = row.get("humannotator_cefr_level", "").strip()
            if not cefr or cefr == "NA":
                cefr = row.get("automarker_cefr_level", "").strip()
            
            if not cefr or cefr == "NA" or cefr not in CEFR_TO_SCORE:
                skipped["no_cefr"] += 1
                continue
            
            # Get split
            split = row.get("split", "").strip().lower()
            if split not in essays:
                skipped["invalid_split"] += 1
                continue
            
            # Get text
            text = row.get("text", "").strip()
            if not text:
                continue
            
            essays[split].append({
                "id": row.get("public_essay_id", ""),
                "text": text,
                "cefr": cefr.upper(),
                "score": CEFR_TO_SCORE[cefr.upper()],
            })
    
    print(f"\nSkipped essays:")
    print(f"  Not final version: {skipped['not_final']}")
    print(f"  No valid CEFR: {skipped['no_cefr']}")
    print(f"  Invalid split: {skipped['invalid_split']}")
    
    return essays


def save_jsonl(data: list[dict], path: Path):
    """Save as JSONL (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            # Format for training: input/target pairs
            f.write(json.dumps({
                "input": item["text"],
                "target": item["score"],
            }, ensure_ascii=False) + "\n")


def print_stats(essays: list[dict], name: str):
    """Print dataset statistics."""
    if not essays:
        print(f"\n{name}: 0 essays")
        return
    
    cefr_counts = Counter(e["cefr"] for e in essays)
    
    print(f"\n{name}:")
    print(f"  Total essays: {len(essays)}")
    print("  CEFR distribution:")
    for cefr in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        count = cefr_counts.get(cefr, 0)
        pct = count / len(essays) * 100 if essays else 0
        bar = "█" * int(pct / 5)  # Simple bar chart
        print(f"    {cefr}: {count:4d} ({pct:5.1f}%) {bar}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare W&I corpus 2024 for training")
    parser.add_argument(
        "--input-dir", 
        required=True, 
        help="Directory containing en-writeandimprove2024-corpus.tsv"
    )
    parser.add_argument(
        "--output-dir", 
        default="data", 
        help="Output directory for JSONL files"
    )
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        print("\nExpected structure:")
        print("  corpus/whole-corpus/en-writeandimprove2024-corpus.tsv")
        return
    
    # Parse corpus (already split into train/dev/test)
    print("=" * 60)
    print("Parsing W&I Corpus 2024")
    print("=" * 60)
    
    essays = parse_wi_corpus_2024(input_dir)
    
    total = sum(len(split) for split in essays.values())
    if total == 0:
        print("\nError: No essays found. Check input directory and file format.")
        return
    
    print(f"\nLoaded {total} essays total (using official train/dev/test splits)")
    
    print_stats(essays["train"], "Training Set")
    print_stats(essays["dev"], "Dev Set")
    print_stats(essays["test"], "Test Set")
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_jsonl(essays["train"], output_dir / "train.jsonl")
    save_jsonl(essays["dev"], output_dir / "dev.jsonl")
    save_jsonl(essays["test"], output_dir / "test.jsonl")
    
    print(f"\n" + "=" * 60)
    print("✅ Data preparation complete!")
    print("=" * 60)
    print(f"Saved to {output_dir}/")
    print(f"   train.jsonl: {len(essays['train']):,} samples")
    print(f"   dev.jsonl:   {len(essays['dev']):,} samples")
    print(f"   test.jsonl:  {len(essays['test']):,} samples")


if __name__ == "__main__":
    main()
