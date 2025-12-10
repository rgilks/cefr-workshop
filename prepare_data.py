"""
Convert W&I corpus to JSONL format for training.

Usage:
    python prepare_data.py --input-dir ./wi+locness --output-dir ./data
"""
import json
from pathlib import Path

# CEFR level to numeric score (1-6 scale)
CEFR_TO_SCORE = {
    "A1": 1.0,
    "A2": 2.0,
    "B1": 3.0,
    "B2": 4.0,
    "C1": 5.0,
    "C2": 6.0,
}


def parse_wi_corpus(input_dir: Path) -> list[dict]:
    """
    Parse the Write & Improve corpus files.
    
    The W&I corpus typically comes with:
    - train.tsv / dev.tsv / test.tsv files
    - Each line: essay_id \t text \t cefr_level
    
    Adjust parsing based on actual file format.
    """
    essays = []
    
    # Try different file patterns
    for pattern in ["*.tsv", "*.csv", "*.txt"]:
        for filepath in input_dir.glob(pattern):
            try:
                with open(filepath, encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split("\t")
                        
                        if len(parts) >= 3:
                            essay_id, text, cefr = parts[0], parts[1], parts[2].upper()
                        elif len(parts) >= 2:
                            # Maybe just text and CEFR
                            text, cefr = parts[0], parts[1].upper()
                            essay_id = f"{filepath.stem}_{line_num}"
                        else:
                            continue
                        
                        # Validate CEFR level
                        if cefr in CEFR_TO_SCORE:
                            essays.append({
                                "id": essay_id,
                                "text": text.strip(),
                                "cefr": cefr,
                                "score": CEFR_TO_SCORE[cefr],
                                "source": filepath.name,
                            })
            except Exception as e:
                print(f"Warning: Could not parse {filepath}: {e}")
    
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
    from collections import Counter
    
    cefr_counts = Counter(e["cefr"] for e in essays)
    
    print(f"\n{name}:")
    print(f"  Total essays: {len(essays)}")
    print("  CEFR distribution:")
    for cefr in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        count = cefr_counts.get(cefr, 0)
        pct = count / len(essays) * 100 if essays else 0
        print(f"    {cefr}: {count:4d} ({pct:5.1f}%)")


def main():
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description="Prepare W&I corpus for training")
    parser.add_argument("--input-dir", required=True, help="Directory with W&I corpus files")
    parser.add_argument("--output-dir", default="data", help="Output directory for JSONL files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    # Parse corpus
    print(f"Parsing W&I corpus from {input_dir}...")
    essays = parse_wi_corpus(input_dir)
    
    if not essays:
        print("Error: No essays found. Check input directory and file format.")
        return
    
    print(f"\nLoaded {len(essays)} essays total")
    print_stats(essays, "All Data")
    
    # Shuffle for random split
    random.shuffle(essays)
    
    # Split: 80% train, 10% dev, 10% test
    n = len(essays)
    n_train = int(0.8 * n)
    n_dev = int(0.1 * n)
    
    train = essays[:n_train]
    dev = essays[n_train:n_train + n_dev]
    test = essays[n_train + n_dev:]
    
    print_stats(train, "Training Set")
    print_stats(dev, "Dev Set")
    print_stats(test, "Test Set")
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(dev, output_dir / "dev.jsonl")
    save_jsonl(test, output_dir / "test.jsonl")
    
    print(f"\n✅ Saved to {output_dir}/")
    print(f"   train.jsonl: {len(train)} samples")
    print(f"   dev.jsonl:   {len(dev)} samples")
    print(f"   test.jsonl:  {len(test)} samples")


if __name__ == "__main__":
    main()
