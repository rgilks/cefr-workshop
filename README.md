# CEFR Scoring with DeBERTa-v3: A Workshop for Programmers

**Duration**: 3-4 hours  
**Prerequisites**: Python basics, command line familiarity  
**Target Audience**: Software engineers new to ML

---

## 🚀 Quick Start (Do This First!)

### Prerequisites

**macOS users** - Install Python 3.12+ if you don't have it:
```bash
# Option 1: Using Homebrew (recommended)
brew install python@3.12

# Option 2: Download from python.org
# https://www.python.org/downloads/
```

**Install uv** - A fast Python package manager:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

### Get the Training Data

**Get the Write & Improve corpus** - this is the real training data:
1. Go to: https://www.cl.cam.ac.uk/research/nl/bea2019st/#data
2. Fill out the request form (access is typically granted immediately)
3. Download and unzip the corpus

### Setup

#### 1. Clone and Install
```bash
git clone <this-repo>
cd cefr-workshop

# Install all dependencies (uv handles Python version automatically)
uv sync
```

#### 2. Verify Your Setup
```bash
# This should print "✅ Model test passed!" (downloads ~400MB model)
uv run python model.py
```

**What this tests**: This verifies that the pre-trained DeBERTa model downloads and loads correctly. You'll see "Sample scores" with random numbers like `[0.31, 0.08]` - these are **meaningless** because the regression head is untrained. After training, scores will be valid CEFR values (1.0-6.0).

#### 3. Prepare the Training Data

**Option A: Using the W&I corpus (recommended)**
```bash
# Once you download and unzip the corpus from Cambridge:
# https://www.cl.cam.ac.uk/research/nl/bea2019st/#data
# 
# The 2024 corpus extracts to a folder like: write-and-improve-corpus-2024-v2/
# Point the script to the whole-corpus subdirectory:

uv run python prepare_data.py \
    --input-dir /path/to/write-and-improve-corpus-2024-v2/whole-corpus \
    --output-dir ./data
```

**Option B: Using synthetic data (for setup testing only)**
```bash
# If you haven't received corpus access yet, you can verify
# the pipeline works with synthetic essays:
uv run python generate_sample_data.py
```

> ⚠️ The synthetic data is only for verifying your setup works. For meaningful model training, use the real W&I corpus.

#### 4. Set Up Modal (Cloud GPUs)
```bash
modal setup        # Follow the prompts to authenticate
uv run modal run hello_modal.py  # Verify Modal works
```

---

## 🎯 What You'll Build

A machine learning model that reads English essays and predicts their CEFR level (A1→C2). By the end, you'll understand:

- How transformer models "read" text
- The training loop (forward pass → loss → backpropagation)
- Running GPU workloads on Modal
- Evaluating model quality with proper metrics

---

## 📁 Project Structure

```
cefr-workshop/
├── README.md                 # This file - workshop guide
├── pyproject.toml            # Python dependencies (used by uv)
├── model.py                  # Model architecture (run to verify setup)
├── train.py                  # Training script for Modal
├── evaluate.py               # Evaluation script
├── serve.py                  # API deployment
├── hello_modal.py            # Quick test for Modal setup
├── prepare_data.py           # Converts W&I corpus to training format
├── generate_sample_data.py   # Creates synthetic test data
└── data/                     # Training data (created by scripts above)
    ├── train.jsonl
    ├── dev.jsonl
    └── test.jsonl
```

---

## 📚 Table of Contents

1. [Understanding the Problem](#part-1-understanding-the-problem-30-min)
2. [The Dataset: Write & Improve Corpus](#part-2-the-dataset-30-min)
3. [How DeBERTa Works (Conceptually)](#part-3-how-deberta-works-45-min)
4. [Setting Up Modal](#part-4-setting-up-modal-15-min)
5. [Building the Training Pipeline](#part-5-building-the-training-pipeline-60-min)
6. [Training & Evaluation](#part-6-training--evaluation-45-min)
7. [Deployment & Inference](#part-7-deployment--inference-30-min)

---

## Part 1: Understanding the Problem (30 min)

### What is CEFR?

The **Common European Framework of Reference for Languages** (CEFR) is the international standard for describing language ability:

| Level | Description | Can Do |
|-------|-------------|--------|
| **A1** | Beginner | Basic phrases, simple questions |
| **A2** | Elementary | Routine tasks, simple descriptions |
| **B1** | Intermediate | Main points, travel situations |
| **B2** | Upper-Intermediate | Complex texts, fluent interaction |
| **C1** | Advanced | Implicit meaning, flexible language |
| **C2** | Mastery | Effortless, precise expression |

### Why Automate This?

- Human scoring is expensive (~$10-50 per essay)
- Bottleneck for language learning platforms
- Models can provide instant feedback

### The ML Framing

This is a **regression problem**:
- **Input**: Essay text (string)
- **Output**: Score 1.0-6.0 (continuous, mapped to CEFR levels)


---

## Part 2: The Dataset (30 min)

### Write & Improve Corpus

The [Write & Improve (W&I) corpus](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data) is from Cambridge English:

- **~2,500 essays** from real English learners (final versions with CEFR labels)
- **Expert CEFR labels** by trained examiners
- **Open access** for research
- Pre-split into train/dev/test sets

### Obtaining the Data

```bash
# 1. Request access from Cambridge
# https://www.cl.cam.ac.uk/research/nl/bea2019st/#data
# Fill out the form - usually approved within 24 hours

# 2. Once you receive the download link, unzip it.
# The 2024 version extracts to: write-and-improve-corpus-2024-v2/
```

### Data Structure

The 2024 corpus is a single TSV file (`whole-corpus/en-writeandimprove2024-corpus.tsv`) with columns:

| Column | Description |
|--------|-------------|
| `text` | The essay text |
| `automarker_cefr_level` | CEFR level from auto-marker |
| `humannotator_cefr_level` | CEFR level from human (if available) |
| `split` | train / dev / test |
| `is_final_version` | TRUE for final drafts |

### CEFR to Numeric Mapping

For regression, we convert CEFR to numbers:

```python
CEFR_TO_SCORE = {
    "A1": 1.0,
    "A2": 2.0,
    "B1": 3.0,
    "B2": 4.0,
    "C1": 5.0,
    "C2": 6.0,
}
```

> ⚠️ **Note**: The W&I corpus primarily contains B1-B2 essays (~75%). Very few A1 or C2 examples exist.

---

## Part 3: How DeBERTa Works (45 min)

### The Transformer Revolution

Before transformers (2017), NLP used recurrent networks that processed words sequentially. Transformers process all words in parallel using **attention**.

```
Traditional (sequential):    The → cat → sat → on → the → mat
Transformer (parallel):      [The, cat, sat, on, the, mat] → processed together
```

### What is DeBERTa-v3?

**DeBERTa** (Decoding-enhanced BERT with disentangled attention) is Microsoft's improved BERT:

| Component | What It Does |
|-----------|--------------|
| **Tokenizer** | Splits text into ~30k subword pieces |
| **Embeddings** | Converts tokens to 1024-dimensional vectors |
| **Encoder** | 24 transformer layers that build meaning |
| **Pooler** | Summarizes the sequence into one vector |

### The Attention Mechanism

Attention answers: "Which words should influence which other words?"

```python
# Simplified attention (pseudocode)
def attention(query, keys, values):
    # How relevant is each key to the query?
    scores = dot_product(query, keys)  
    weights = softmax(scores)  # Normalize to sum to 1
    return weighted_sum(values, weights)
```

**Example**: For "The bank by the river", when processing "bank":
- High attention to "river" → suggests "riverbank", not "financial bank"

### Pre-training vs Fine-tuning

1. **Pre-training** (done by Microsoft):
   - Trained on billions of words
   - Task: Predict masked words ("The [MASK] sat on the mat" → "cat")
   - Result: General language understanding

2. **Fine-tuning** (what we'll do):
   - Start with pre-trained weights
   - Train on our CEFR essays
   - Only need ~3,000 examples!


### Recommended Reading (15 min)

Before continuing, skim:
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual guide
- [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) - Applies to DeBERTa

---

## Part 4: Setting Up Modal (15 min)

### What is Modal?

Modal is a cloud platform for running Python functions on GPUs. Think of it as "serverless for ML":

```python
@app.function(gpu="A10G")
def train():
    # This runs on a GPU in the cloud
    pass
```

### Installation

```bash
# 1. Install Modal
pip install modal

# 2. Authenticate (creates ~/.modal.toml)
modal setup

# 3. Verify
modal run --help
```

### Modal Concepts

| Concept | Description |
|---------|-------------|
| **App** | Container for your functions |
| **Image** | Docker-like environment specification |
| **Volume** | Persistent storage across runs |
| **Secret** | Secure credential storage |

### Your First Modal Script

Create `hello_modal.py`:

```python
import modal

app = modal.App("hello-workshop")

@app.function()
def hello():
    import platform
    return f"Hello from {platform.node()}!"

@app.local_entrypoint()
def main():
    print(hello.remote())
```

Run it:
```bash
modal run hello_modal.py
# Output: Hello from modal-runner-xxx!
```

---

## Part 5: Building the Training Pipeline (60 min)

> 📁 See the [Project Structure](#-project-structure) at the top of this document for file organization.

> 💡 **Note**: The code snippets below are **simplified for teaching**. The actual project files (`prepare_data.py`, `model.py`, `train.py`) include additional features like better error handling, statistics output, and support for the 2024 corpus format. Feel free to read both!

### Step 1: Data Preparation Script

The actual `prepare_data.py` handles the 2024 corpus format. Here's a simplified version showing the core logic:

```python
"""
Convert W&I corpus to JSONL format for training.
"""
import json
from pathlib import Path

# CEFR level to numeric score
CEFR_TO_SCORE = {
    "A1": 1.0, "A2": 2.0, "B1": 3.0,
    "B2": 4.0, "C1": 5.0, "C2": 6.0,
}

def parse_wi_corpus(input_dir: Path) -> list[dict]:
    """Parse the W&I corpus files."""
    essays = []
    
    # The corpus has separate files for each level
    for tsv_file in input_dir.glob("*.tsv"):
        with open(tsv_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    essay_id, text, cefr = parts[0], parts[1], parts[2]
                    if cefr in CEFR_TO_SCORE:
                        essays.append({
                            "id": essay_id,
                            "text": text,
                            "cefr": cefr,
                            "score": CEFR_TO_SCORE[cefr],
                        })
    
    return essays

def save_jsonl(data: list[dict], path: Path):
    """Save as JSONL (one JSON object per line)."""
    with open(path, "w") as f:
        for item in data:
            # Format for training: input/target pairs
            f.write(json.dumps({
                "input": item["text"],
                "target": item["score"],
            }) + "\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()
    
    essays = parse_wi_corpus(Path(args.input_dir))
    print(f"Loaded {len(essays)} essays")
    
    # Split: 80% train, 10% dev, 10% test
    # (In practice, use the official splits from the corpus)
    n = len(essays)
    train = essays[:int(0.8 * n)]
    dev = essays[int(0.8 * n):int(0.9 * n)]
    test = essays[int(0.9 * n):]
    
    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)
    
    save_jsonl(train, out / "train.jsonl")
    save_jsonl(dev, out / "dev.jsonl")
    save_jsonl(test, out / "test.jsonl")
    
    print(f"Saved: {len(train)} train, {len(dev)} dev, {len(test)} test")
```

### Step 2: Model Architecture

Create `model.py`:

```python
"""
CEFR scoring model based on DeBERTa-v3.

Architecture:
    DeBERTa-v3-base Encoder (86M params)
        ↓
    Mean Pooling (average all token representations)
        ↓
    Regression Head (Linear → ReLU → Linear → 1)
        ↓
    CEFR Score (1.0 - 6.0)
"""
import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config


class CEFRModel(nn.Module):
    """
    Fine-tuned DeBERTa for CEFR score prediction.
    
    Why this architecture?
    - DeBERTa-v3-base: Good balance of quality vs speed
    - Mean pooling: More robust than [CLS] token alone
    - Simple regression head: Prevents overfitting on small datasets
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Load pre-trained encoder
        self.encoder = DebertaV2Model.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for base
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: 1 for real tokens, 0 for padding [batch_size, seq_len]
        
        Returns:
            Predicted scores [batch_size]
        """
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # [batch, seq, hidden]
        
        # Mean pooling: average non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch, seq, 1]
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)  # [batch, hidden]
        count = mask_expanded.sum(dim=1).clamp(min=1e-9)  # [batch, 1]
        pooled = sum_hidden / count  # [batch, hidden]
        
        # Predict score
        score = self.regressor(pooled).squeeze(-1)  # [batch]
        
        return score


def score_to_cefr(score: float) -> str:
    """Convert numeric score to CEFR level."""
    if score < 1.5:
        return "A1"
    elif score < 2.5:
        return "A2"
    elif score < 3.5:
        return "B1"
    elif score < 4.5:
        return "B2"
    elif score < 5.5:
        return "C1"
    else:
        return "C2"
```

### Step 3: Training Script

Create `train.py`:

```python
"""
Train CEFR scoring model on Modal.

Usage:
    modal run train.py              # Full training
    modal run train.py --test-run   # Quick test (5 min)
"""
import json
import os
from pathlib import Path

import modal

# ============================================================
# Modal Configuration
# ============================================================

app = modal.App("cefr-workshop")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "datasets>=2.14.0",
        "scikit-learn>=1.3.0",
        "sentencepiece>=0.1.99",  # Required for DeBERTa tokenizer
    )
    # Add our code
    .add_local_file("model.py", "/app/model.py")
    .add_local_dir("data", "/app/data")
)

# Persistent storage for trained models
volume = modal.Volume.from_name("cefr-models", create_if_missing=True)


# ============================================================
# Training Function
# ============================================================

@app.function(
    image=image,
    gpu="A10G",  # NVIDIA A10G: 24GB VRAM, good price/performance
    timeout=3600,  # 1 hour max
    volumes={"/vol": volume},
)
def train(
    test_run: bool = False,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    num_epochs: int = 10,
    max_length: int = 512,
):
    """
    Train the CEFR model.
    
    Args:
        test_run: If True, train for 1 epoch on small subset
        learning_rate: AdamW learning rate (2e-5 is standard for fine-tuning)
        batch_size: Samples per gradient update
        num_epochs: Full passes through training data
        max_length: Max tokens (512 captures most essays)
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from sklearn.metrics import mean_absolute_error
    
    # Import our model
    import sys
    sys.path.insert(0, "/app")
    from model import CEFRModel, score_to_cefr
    
    print("=" * 60)
    print("CEFR Model Training")
    print("=" * 60)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Test run: {test_run}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs if not test_run else 1}")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --------------------------------------------------------
    # Load Data
    # --------------------------------------------------------
    
    def load_jsonl(path: str) -> list[dict]:
        with open(path) as f:
            return [json.loads(line) for line in f]
    
    train_data = load_jsonl("/app/data/train.jsonl")
    dev_data = load_jsonl("/app/data/dev.jsonl")
    
    if test_run:
        train_data = train_data[:100]
        dev_data = dev_data[:50]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(dev_data)}")
    
    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    
    class CEFRDataset(Dataset):
        def __init__(self, data: list[dict]):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            encoding = tokenizer(
                item["input"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(item["target"], dtype=torch.float32),
            }
    
    train_loader = DataLoader(
        CEFRDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
    )
    dev_loader = DataLoader(
        CEFRDataset(dev_data),
        batch_size=batch_size,
    )
    
    # --------------------------------------------------------
    # Model Setup
    # --------------------------------------------------------
    
    model = CEFRModel().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer: AdamW with weight decay (regularization)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    
    # Learning rate scheduler: linear warmup then decay
    num_training_steps = len(train_loader) * (1 if test_run else num_epochs)
    num_warmup_steps = num_training_steps // 10
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Loss function: Mean Squared Error
    loss_fn = torch.nn.MSELoss()
    
    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------
    
    best_dev_mae = float("inf")
    epochs = 1 if test_run else num_epochs
    
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            predictions = model(input_ids, attention_mask)
            loss = loss_fn(predictions, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]
                
                predictions = model(input_ids, attention_mask)
                
                all_preds.extend(predictions.cpu().tolist())
                all_labels.extend(labels.tolist())
        
        dev_mae = mean_absolute_error(all_labels, all_preds)
        
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Dev MAE: {dev_mae:.4f}")
        
        # Save best model
        if dev_mae < best_dev_mae:
            best_dev_mae = dev_mae
            torch.save(model.state_dict(), "/vol/best_model.pt")
            tokenizer.save_pretrained("/vol/tokenizer")
            print(f"  ✅ New best model saved!")
    
    # --------------------------------------------------------
    # Final Evaluation
    # --------------------------------------------------------
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Dev MAE: {best_dev_mae:.4f}")
    print("Model saved to /vol/best_model.pt")
    print("=" * 60)
    
    # Commit volume to persist
    volume.commit()
    
    return {"best_dev_mae": best_dev_mae}


# ============================================================
# Entry Point
# ============================================================

@app.local_entrypoint()
def main(test_run: bool = False):
    """Run training."""
    result = train.remote(test_run=test_run)
    print(f"Result: {result}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true")
    args = parser.parse_args()
    
    with app.run():
        train.remote(test_run=args.test_run)
```

---

## Part 6: Training & Evaluation (45 min)

### Running Training

```bash
# Quick test (verifies everything works, ~5 min)
modal run train.py --test-run

# Full training (~30-60 min)
modal run train.py
```

### Understanding the Output

```
Epoch 1/10
  Train Loss: 0.8234
  Dev MAE: 0.52
  ✅ New best model saved!

Epoch 2/10
  Train Loss: 0.4123
  Dev MAE: 0.41
  ✅ New best model saved!
  
...
```

**Key Metrics**:

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **Train Loss** | MSE on training data | Decreasing |
| **Dev MAE** | Mean Absolute Error on validation | < 0.5 |

### What is MAE?

**Mean Absolute Error** = average of `|predicted - actual|`

```python
# Example:
predictions = [3.2, 4.1, 2.8]
actuals     = [3.0, 4.0, 3.0]
errors      = [0.2, 0.1, 0.2]
MAE = average(errors) = 0.167
```

An MAE of 0.4 means predictions are within half a CEFR level on average.

### Evaluation Script

Create `evaluate.py`:

```python
"""
Evaluate trained model on test set.

Usage:
    modal run evaluate.py
"""
import json
import modal
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
import numpy as np

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
    from transformers import  AutoTokenizer
    
    import sys
    sys.path.insert(0, "/app")
    from model import CEFRModel, score_to_cefr
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = CEFRModel()
    model.load_state_dict(torch.load("/vol/best_model.pt"))
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
        for item in test_data:
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
    
    # Metrics
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
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"MAE:               {mae:.3f}")
    print(f"QWK:               {qwk:.3f}")
    print(f"Exact Accuracy:    {exact_match:.1%}")
    print(f"Adjacent Accuracy: {adjacent:.1%}")
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
    print(f"Result: {result}")
```

### Understanding QWK (Quadratic Weighted Kappa)

QWK measures agreement between predicted and actual CEFR levels:

| QWK Value | Interpretation |
|-----------|----------------|
| 1.0 | Perfect agreement |
| 0.8-1.0 | **Excellent** (human-level) |
| 0.6-0.8 | Good |
| 0.4-0.6 | Moderate |
| < 0.4 | Needs improvement |

> 💡 **Why QWK?** Unlike accuracy, QWK penalizes larger errors more. Predicting C2 when the actual is A2 is much worse than predicting B2.

---

## Part 7: Deployment & Inference (30 min)

### Creating an API

The `serve.py` file deploys your trained model as a REST API:

```python
"""
Serve CEFR model as a FastAPI endpoint on Modal.

Usage:
    modal deploy serve.py
    # Then: curl https://your-app.modal.run/score -X POST -d '{"text": "..."}'
"""
import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = modal.App("cefr-api")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "fastapi>=0.104.0",
        "sentencepiece>=0.1.99",
    )
    .add_local_file("model.py", "/app/model.py")
)

volume = modal.Volume.from_name("cefr-models")

# FastAPI app
web_app = FastAPI(title="CEFR Scoring API")


class ScoreRequest(BaseModel):
    text: str


class ScoreResponse(BaseModel):
    score: float
    cefr_level: str
    confidence: str


@app.cls(
    image=image,
    gpu="T4",
    volumes={"/vol": volume},
    scaledown_window=60,  # Keep warm for 1 minute
)
class CEFRService:
    """CEFR scoring service with model lifecycle management."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
    
    @modal.enter()
    def startup(self):
        """Load model on container startup."""
        import torch
        from transformers import AutoTokenizer
        import sys
        sys.path.insert(0, "/app")
        from model import CEFRModel
        
        print("Loading model...")
        self.model = CEFRModel()
        self.model.load_state_dict(torch.load("/vol/best_model.pt", map_location="cpu"))
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained("/vol/tokenizer")
        print("Model loaded!")
    
    @modal.asgi_app()
    def serve(self):
        """Return the FastAPI app."""
        
        @web_app.get("/health")
        def health():
            return {"status": "healthy", "model_loaded": self.model is not None}
        
        @web_app.post("/score", response_model=ScoreResponse)
        def score_essay(request: ScoreRequest):
            """Score an essay and return CEFR level."""
            import torch
            import sys
            sys.path.insert(0, "/app")
            from model import score_to_cefr
            
            if self.model is None:
                raise HTTPException(500, "Model not loaded")
            
            if len(request.text.strip()) < 10:
                raise HTTPException(400, "Text too short (min 10 characters)")
            
            # Tokenize
            encoding = self.tokenizer(
                request.text,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )
            
            # Predict
            with torch.no_grad():
                score = self.model(
                    encoding["input_ids"],
                    encoding["attention_mask"],
                ).item()
            
            # Clamp to valid range
            score = max(1.0, min(6.0, score))
            cefr = score_to_cefr(score)
            
            # Simple confidence based on distance to CEFR boundaries
            boundaries = {1.5, 2.5, 3.5, 4.5, 5.5}
            min_dist = min(abs(score - b) for b in boundaries)
            confidence = "high" if min_dist > 0.3 else "medium" if min_dist > 0.15 else "low"
            
            return ScoreResponse(
                score=round(score, 2),
                cefr_level=cefr,
                confidence=confidence,
            )
        
        return web_app
```

### Deploying

```bash
# Deploy (gets a persistent URL)
modal deploy serve.py

# Output:
# ✓ Created CEFR-API
# ✓ https://your-username--cefr-api-cefrservice-serve.modal.run
```

### Testing the API

```bash
# Score an essay
curl -X POST \
  https://your-app.modal.run/score \
  -H "Content-Type: application/json" \
  -d '{"text": "I think technology is very important in our life today. Many people use smartphones and computers every day. In my opinion, technology helps us communicate with friends and family."}'

# Response:
{
  "score": 3.42,
  "cefr_level": "B1",
  "confidence": "medium"
}
```

---

## 📖 Reference Materials

### Essential Reading (Do These First)

1. **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)** - Visual guide to attention (30 min)
2. **[The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)** - How pre-training works (20 min)
3. **[But what is a GPT?](https://www.youtube.com/watch?v=wjZofJX0v4M)** - 3Blue1Brown video (27 min)

### Technical Documentation

- [Hugging Face Transformers](https://huggingface.co/docs/transformers) - Library docs
- [Modal Documentation](https://modal.com/docs/guide) - Cloud platform
- [DeBERTa Paper](https://arxiv.org/abs/2006.03654) - Original research
- [W&I Corpus](https://www.cl.cam.ac.uk/research/nl/bea2019st/) - Dataset source

### CEFR Background

- [CEFR Overview](https://www.coe.int/en/web/common-european-framework-reference-languages) - Official Council of Europe
- [CEFR Level Descriptors](https://www.cambridgeenglish.org/exams-and-tests/cefr/) - Cambridge English


## 🎯 Workshop Exercises

### Exercise 1: Data Exploration (15 min)

Write a script to:
1. Load the training data
2. Print the distribution of CEFR levels
3. Find the shortest and longest essays

### Exercise 2: Hyperparameter Tuning (30 min)

Try different configurations and compare results:
- Learning rate: `1e-5`, `2e-5`, `5e-5`
- Batch size: `8`, `16`, `32`
- Epochs: `5`, `10`, `15`

### Exercise 3: Error Analysis (20 min)

After training:
1. Find the 10 worst predictions (highest error)
2. Read those essays - why might the model struggle?
3. Are there patterns in the errors?

### Exercise 4: Ensemble Model (Advanced, 45 min)

Train 3 models with different random seeds, then average their predictions:
```python
final_score = (model1(text) + model2(text) + model3(text)) / 3
```

Does this improve MAE?

---

## ❓ FAQ

**Q: How long does training take?**  
A: ~30-60 minutes for 10 epochs on an A10G GPU.

**Q: What if I don't have a GPU?**  
A: Modal provides cloud GPUs. The free tier includes ~30 GPU-hours/month.

**Q: Can I use a different model?**  
A: Yes! Try `microsoft/deberta-v3-small` (faster) or `microsoft/deberta-v3-large` (better).

**Q: Why DeBERTa over BERT/RoBERTa?**  
A: DeBERTa-v3 consistently outperforms on NLU benchmarks while being similar in size.

**Q: What's the cost?**  
A: ~$0.50-1.00 for a full training run on Modal (A10G at $1.10/hr).

---

## 🏁 Next Steps

After completing this workshop:

1. **Try larger models**: DeBERTa-v3-large has 304M params (vs 86M for base)
2. **Multi-task learning**: Predict multiple dimensions (grammar, vocabulary, etc.)
3. **Data augmentation**: Paraphrase essays to increase training data
4. **Ordinal regression**: Use CORN loss for better CEFR boundary handling

Happy training! 🎓
