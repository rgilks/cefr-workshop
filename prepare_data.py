"""
Convert the PELIC dataset to JSONL format for training.

Usage:
    uv run python prepare_data.py
"""

import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# CEFR level to numeric score (1-6 scale)
CEFR_TO_SCORE = {
    "A1": 1.0, "A2": 2.0, "B1": 3.0, "B2": 4.0, "C1": 5.0, "C2": 6.0
}

# PELIC Level Mapping
# Level 2 -> A2, Level 3 -> B1, Level 4 -> B2, Level 5 -> C1
PELIC_TO_CEFR = {
    "2": "A2", "3": "B1", "4": "B2", "5": "C1"
}


def load_pelic(output_dir: Path) -> Dict[str, List[dict]]:
    """Clone and load the PELIC dataset."""
    try:
        import pandas as pd
    except ImportError:
        logger.error("Pandas is required. Run 'uv sync' to install dependencies.")
        sys.exit(1)

    repo_url = "https://github.com/eli-data-mining-group/PELIC-dataset.git"
    target_dir = output_dir.parent / "pelic_raw"
    
    if not target_dir.exists():
        logger.info(f"Cloning PELIC dataset to {target_dir}...")
        try:
            import subprocess
            subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target_dir)], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone PELIC: {e}")
            sys.exit(1)
            
    csv_path = target_dir / "PELIC_compiled.csv"
    if not csv_path.exists():
        logger.error(f"Could not find PELIC_compiled.csv in {target_dir}")
        sys.exit(1)
        
    logger.info(f"Parsing PELIC CSV: {csv_path}...")
    
    # Check if file is small (LFS pointer)
    if csv_path.stat().st_size < 1000:
        logger.warning("PELIC CSV seems to be a Git LFS pointer. Attempting to pull...")
        import subprocess
        subprocess.run(["git", "lfs", "pull"], cwd=target_dir, check=False)
    
    try:
        df = pd.read_csv(csv_path, usecols=["answer_id", "text", "level_id", "version"])
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        sys.exit(1)
        
    # Filter to version 1 (original submissions)
    df = df[df['version'] == 1]
    
    data_list = []
    
    for _, row in df.iterrows():
        text = str(row['text']).strip()
        lvl = str(row['level_id'])
        
        if lvl not in PELIC_TO_CEFR:
            continue
            
        if len(text) < 10:  # Skip noise
            continue
            
        cefr = PELIC_TO_CEFR[lvl]
        data_list.append({
            "text": text,
            "cefr": cefr,
            "score": CEFR_TO_SCORE[cefr]
        })
        
    logger.info(f"Loaded {len(data_list)} essays from PELIC.")
    
    # Shuffle and split 80/10/10
    random.seed(42)
    random.shuffle(data_list)
    n = len(data_list)
    n_train = int(n * 0.8)
    n_dev = int(n * 0.1)
    
    return {
        "train": data_list[:n_train],
        "dev": data_list[n_train:n_train+n_dev],
        "test": data_list[n_train+n_dev:]
    }


def save_jsonl(data: list[dict], path: Path):
    """Save as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
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
    for cefr in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        count = cefr_counts.get(cefr, 0)
        pct = count / len(essays) * 100 if essays else 0
        bar = "█" * int(pct / 5)
        print(f"    {cefr}: {count:4d} ({pct:5.1f}%) {bar}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare PELIC dataset for training")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    essays = load_pelic(output_dir)

    total = sum(len(e) for e in essays.values())
    if total == 0:
        logger.error("No essays loaded.")
        return

    print_stats(essays.get("train", []), "Training Set")
    print_stats(essays.get("dev", []), "Dev Set")
    print_stats(essays.get("test", []), "Test Set")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(essays.get("train", []), output_dir / "train.jsonl")
    save_jsonl(essays.get("dev", []), output_dir / "dev.jsonl")
    save_jsonl(essays.get("test", []), output_dir / "test.jsonl")
    
    print(f"\n✅ Data saved to {output_dir}/")


if __name__ == "__main__":
    main()
