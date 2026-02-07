"""
CEFR scoring model based on DeBERTa-v3.

Architecture:
    DeBERTa-v3-base Encoder (~184M params)
        ↓
    Mean Pooling (average all token representations)
        ↓
    Regression Head (Linear → ReLU → Linear → 1)
        ↓
    CEFR Score (1.0 - 6.0)
"""
import torch
import torch.nn as nn
from transformers import DebertaV2Model


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
        # NOTE: This is randomly initialized! It will output meaningless scores
        # until you train the model with train.py. After training, it learns to
        # map DeBERTa's language understanding to CEFR scores (1.0-6.0).
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


# CEFR level to numeric score mapping (single source of truth)
CEFR_TO_SCORE = {
    "A1": 1.0, "A2": 2.0, "B1": 3.0,
    "B2": 4.0, "C1": 5.0, "C2": 6.0,
}


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


if __name__ == "__main__":
    # Quick test to verify the model architecture loads correctly.
    # NOTE: The scores produced here are RANDOM because the regression head
    # is untrained. Valid CEFR scores (1.0-6.0) only appear after training.
    print("Testing CEFRModel...")
    print("(Note: Scores will be random until you train the model)\n")
    model = CEFRModel()
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
    
    # Test forward pass with dummy input
    dummy_ids = torch.randint(0, 1000, (2, 128))
    dummy_mask = torch.ones(2, 128)
    
    with torch.no_grad():
        scores = model(dummy_ids, dummy_mask)
    
    print(f"Output shape: {scores.shape}")
    print(f"Sample scores: {scores.tolist()}")
    print("✅ Model test passed!")
