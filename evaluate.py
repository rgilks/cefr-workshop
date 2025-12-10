"""
Evaluate trained model on test set.

Usage:
    modal run evaluate.py
"""
import json
import modal

app = modal.App("cefr-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "scikit-learn>=1.3.0",
        "sentencepiece>=0.1.99",
    )
    .add_local_file("model.py", "/app/model.py")
    .add_local_dir("data", "/app/data")
)

volume = modal.Volume.from_name("cefr-models")


@app.function(
    image=image,
    gpu="T4",  # Smaller GPU is fine for inference
    volumes={"/vol": volume},
)
def evaluate():
    """Evaluate model on held-out test set."""
    import torch
    from transformers import AutoTokenizer
    from sklearn.metrics import mean_absolute_error, cohen_kappa_score
    
    import sys
    sys.path.insert(0, "/app")
    from model import CEFRModel, score_to_cefr
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print("Loading model...")
    model = CEFRModel()
    model.load_state_dict(torch.load("/vol/best_model.pt", map_location=device))
    model = model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("/vol/tokenizer")
    
    # Load test data
    with open("/app/data/test.jsonl") as f:
        test_data = [json.loads(line) for line in f]
    
    print(f"Evaluating on {len(test_data)} test samples...")
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for i, item in enumerate(test_data):
            encoding = tokenizer(
                item["input"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )
            
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            pred = model(input_ids, attention_mask).item()
            predictions.append(pred)
            actuals.append(item["target"])
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(test_data)}")
    
    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    
    # Convert to CEFR levels for QWK
    pred_cefr = [score_to_cefr(p) for p in predictions]
    actual_cefr = [score_to_cefr(a) for a in actuals]
    
    # Map CEFR to integers for kappa
    cefr_to_int = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
    pred_int = [cefr_to_int[c] for c in pred_cefr]
    actual_int = [cefr_to_int[c] for c in actual_cefr]
    
    qwk = cohen_kappa_score(actual_int, pred_int, weights="quadratic")
    
    # Accuracy metrics
    exact_match = sum(p == a for p, a in zip(pred_cefr, actual_cefr)) / len(actuals)
    adjacent = sum(
        abs(cefr_to_int[p] - cefr_to_int[a]) <= 1 
        for p, a in zip(pred_cefr, actual_cefr)
    ) / len(actuals)
    
    # Per-level analysis
    from collections import defaultdict
    level_errors = defaultdict(list)
    for pred, actual, actual_cefr_label in zip(predictions, actuals, actual_cefr):
        level_errors[actual_cefr_label].append(abs(pred - actual))
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples:           {len(test_data)}")
    print(f"MAE:               {mae:.3f}")
    print(f"QWK:               {qwk:.3f}")
    print(f"Exact Accuracy:    {exact_match:.1%}")
    print(f"Adjacent Accuracy: {adjacent:.1%}")
    print("\nPer-Level MAE:")
    for cefr in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        if level_errors[cefr]:
            level_mae = sum(level_errors[cefr]) / len(level_errors[cefr])
            print(f"  {cefr}: {level_mae:.3f} (n={len(level_errors[cefr])})")
    print("=" * 60)
    
    return {
        "mae": mae,
        "qwk": qwk,
        "exact_accuracy": exact_match,
        "adjacent_accuracy": adjacent,
    }


@app.local_entrypoint()
def main():
    result = evaluate.remote()
    print(f"\nFinal Result: {result}")
