"""
autoencoder.py
--------------
Symmetric autoencoder for feature extraction / dimensionality reduction.
Architecture kept consistent with base paper (5 layers, latent=50)
but trained on GAN-augmented data for better representation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 80, latent_dim: int = 50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 80, output_dim: int = 122):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class Autoencoder(nn.Module):
    """
    Symmetric autoencoder (mirrors discriminator in BEGAN architecture).
    Encoder is used as feature extractor — frozen during classifier training.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 80, latent_dim: int = 50):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)

    def reconstruction_loss(self, x, x_hat):
        return nn.functional.mse_loss(x_hat, x)


class AutoencoderTrainer:
    """
    Trains the autoencoder on the GAN-augmented dataset.
    Stops early when reconstruction accuracy > target_acc.

    Usage:
        ae = AutoencoderTrainer(input_dim=122)
        ae.fit(X_augmented)
        encoder = ae.get_encoder()   # frozen encoder for classifiers
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 80,
        latent_dim: int = 50,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = Autoencoder(input_dim, hidden_dim, latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_losses = []

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 300,
        batch_size: int = 256,
        target_acc: float = 0.97,
        patience: int = 35,
        verbose: bool = True,
    ):
        X_t = torch.FloatTensor(X).to(self.device)
        loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            epoch_losses = []
            for (x,) in loader:
                x = x.to(self.device)
                x_hat, _ = self.model(x)
                loss = self.model.reconstruction_loss(x, x_hat)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_loss)

            # approximate reconstruction accuracy
            recon_acc = 1.0 - avg_loss
            if verbose and epoch % 50 == 0:
                print(f"AE Epoch {epoch:4d}/{epochs} | Loss: {avg_loss:.5f} | Recon acc: {recon_acc:.4f}")

            if recon_acc >= target_acc:
                print(f"[AE] Early stop at epoch {epoch} — recon acc {recon_acc:.4f} >= {target_acc}")
                break

            # patience check
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[AE] Early stop at epoch {epoch} — no improvement for {patience} epochs.")
                    break

    def get_encoder(self) -> Encoder:
        """Returns frozen encoder module for use as feature extractor."""
        encoder = self.model.encoder
        for param in encoder.parameters():
            param.requires_grad = False
        return encoder

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode X into latent space."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            z = self.model.encode(X_t).cpu().numpy()
        self.model.train()
        return z

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
