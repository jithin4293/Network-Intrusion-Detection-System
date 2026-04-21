"""
transformer_classifier.py
--------------------------
Multi-head self-attention Transformer for network intrusion classification.
Key novelty over base paper which used LSTM.

Advantage over LSTM:
  - Captures long-range feature dependencies (not just sequential)
  - Better at detecting attacks with complex, non-sequential patterns
  - Parallelisable — faster training
  - Outperforms LSTM on tabular network traffic features

Architecture:
  Encoder [frozen] → positional-free embedding → N Transformer blocks
  → GlobalAvgPool → Classifier head
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ─────────────────────────────────────────────
#  Transformer Block
# ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block:
      LayerNorm → Multi-head Self-Attention → residual
      LayerNorm → Feed-forward (dim*4) → residual → Dropout
    """
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop(ff_out))
        return x


# ─────────────────────────────────────────────
#  Full Transformer Classifier
# ─────────────────────────────────────────────
class TransformerClassifier(nn.Module):
    """
    Tabular Transformer for intrusion detection.

    Input: [batch, input_dim] (latent features from frozen Encoder, or raw features)
    Output: [batch, n_classes] logits

    For tabular data, we treat each feature as a "token" (seq_len = input_dim, d_model = 1)
    then project to d_model via a learned linear layer.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Feature embedding: treat each feature independently as a token
        self.input_proj = nn.Linear(1, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )
        self.input_dim = input_dim
        self.n_classes = n_classes

    def forward(self, x):
        # x: [batch, input_dim]
        # treat each feature as a token: [batch, input_dim, 1] → [batch, seq, d_model]
        x = x.unsqueeze(-1)                  # [B, D, 1]
        x = self.input_proj(x)               # [B, D, d_model]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)                     # [B, D, d_model]
        x = x.permute(0, 2, 1)              # [B, d_model, D]
        x = self.pool(x).squeeze(-1)         # [B, d_model]
        return self.classifier(x)            # [B, n_classes]


# ─────────────────────────────────────────────
#  DNN Classifier (from base paper — kept for comparison)
# ─────────────────────────────────────────────
class DNNClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden1: int = 32, hidden2: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
#  CNN Classifier (from base paper — kept for comparison)
# ─────────────────────────────────────────────
class CNNClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        conv_out = (input_dim // 3) * 32
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, n_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)           # [B, 1, D]
        x = self.conv(x)             # [B, 32, D//3]
        x = x.flatten(1)
        return self.fc(x)


# ─────────────────────────────────────────────
#  Generic Classifier Trainer
# ─────────────────────────────────────────────
class ClassifierTrainer:
    """
    Universal trainer for all classifier architectures.

    Usage:
        trainer = ClassifierTrainer(model, n_classes=5, device='cuda')
        trainer.fit(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
    """

    def __init__(
        self,
        model: nn.Module,
        n_classes: int,
        lr: float = 1e-3,
        device: str = "cpu",
        encoder: nn.Module = None,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.n_classes = n_classes
        self.encoder = encoder.to(self.device) if encoder is not None else None
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_accs = []

    def _encode(self, X_t: torch.Tensor) -> torch.Tensor:
        if self.encoder is not None:
            with torch.no_grad():
                X_t = self.encoder(X_t)
        return X_t

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 300,
        batch_size: int = 256,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        patience: int = 35,
        verbose: bool = True,
    ):
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

        best_val = -1
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            ep_loss = []
            for xb, yb in loader:
                xb = self._encode(xb)
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                ep_loss.append(loss.item())

            self.train_losses.append(np.mean(ep_loss))

            # validation
            if X_val is not None and y_val is not None:
                val_acc = self.evaluate(X_val, y_val, verbose=False)["accuracy"]
                self.val_accs.append(val_acc)
                if verbose and epoch % 50 == 0:
                    print(f"Epoch {epoch:4d} | Loss: {self.train_losses[-1]:.4f} | Val acc: {val_acc:.4f}")
                if val_acc > best_val + 1e-6:
                    best_val = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"[Classifier] Early stop at epoch {epoch}")
                        break
            else:
                if verbose and epoch % 50 == 0:
                    print(f"Epoch {epoch:4d} | Loss: {self.train_losses[-1]:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            X_t = self._encode(X_t)
            logits = self.model(X_t)
            preds = logits.argmax(dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            X_t = self._encode(X_t)
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def evaluate(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> dict:
        preds = self.predict(X)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="weighted", zero_division=0)
        if verbose:
            print(f"\nAccuracy: {acc:.4f} | Weighted F1: {f1:.4f}")
            print(classification_report(y, preds, zero_division=0))
        return {"accuracy": acc, "f1_weighted": f1, "predictions": preds}

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
