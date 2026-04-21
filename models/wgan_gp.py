"""
wgan_gp.py
----------
Wasserstein GAN with Gradient Penalty (WGAN-GP)
Replaces BEGAN from base paper for more stable training
and better minority class synthesis.

Reference: Gulrajani et al., "Improved Training of Wasserstein GANs" (2017)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────
#  Generator
# ─────────────────────────────────────────────
class Generator(nn.Module):
    """
    Maps latent noise z → synthetic network traffic features.
    Architecture mirrors the decoder of the base paper's BEGAN generator,
    but with LayerNorm for WGAN-GP stability.
    """
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),          # output in [0,1] (min-max scaled features)
        )

    def forward(self, z):
        return self.net(z)


# ─────────────────────────────────────────────
#  Critic (Discriminator for WGAN)
# ─────────────────────────────────────────────
class Critic(nn.Module):
    """
    Outputs a real-valued score (not a probability).
    No sigmoid at output — required for WGAN-GP.
    Spectral norm NOT used here; gradient penalty handles Lipschitz constraint.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
#  WGAN-GP Trainer
# ─────────────────────────────────────────────
class WGANGPTrainer:
    """
    Trains one Generator+Critic pair per attack class.
    After training, call generate(n) to produce n synthetic samples.

    Key improvement over base paper:
        - WGAN-GP is more stable than BEGAN for tabular data
        - Gradient penalty enforces Lipschitz constraint directly
        - No mode collapse issues that affect BEGAN on rare classes

    Parameters:
        latent_dim   : size of noise vector (default 64)
        hidden_dim   : hidden layer width (default 256)
        n_critic     : critic updates per generator update (default 5)
        lambda_gp    : gradient penalty weight (default 10)
        lr           : Adam learning rate (default 1e-4)
        device       : 'cuda' or 'cpu'
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        n_critic: int = 5,
        lambda_gp: float = 10.0,
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.device = torch.device(device)

        self.G = Generator(latent_dim, input_dim, hidden_dim).to(self.device)
        self.C = Critic(input_dim, hidden_dim).to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=lr, betas=(0.0, 0.9))
        self.opt_C = optim.Adam(self.C.parameters(), lr=lr, betas=(0.0, 0.9))

        self.g_losses = []
        self.c_losses = []
        self.w_distances = []

    # ── training ────────────────────────────────
    def fit(self, X: np.ndarray, epochs: int = 300, batch_size: int = 256,
            verbose: bool = True, verbose_every: int = 50):
        """
        Train WGAN-GP on class-specific data X (already preprocessed, shape [N, D]).
        """
        X_t = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(1, epochs + 1):
            g_loss_epoch, c_loss_epoch, wd_epoch = [], [], []

            for (real,) in loader:
                real = real.to(self.device)
                bs = real.size(0)

                # ── train critic n_critic times ──
                for _ in range(self.n_critic):
                    z = torch.randn(bs, self.latent_dim, device=self.device)
                    fake = self.G(z).detach()

                    c_real = self.C(real)
                    c_fake = self.C(fake)
                    gp = self._gradient_penalty(real, fake)

                    c_loss = c_fake.mean() - c_real.mean() + self.lambda_gp * gp
                    self.opt_C.zero_grad()
                    c_loss.backward()
                    self.opt_C.step()

                    c_loss_epoch.append(c_loss.item())
                    wd_epoch.append((c_real.mean() - c_fake.mean()).item())

                # ── train generator once ──
                z = torch.randn(bs, self.latent_dim, device=self.device)
                fake = self.G(z)
                g_loss = -self.C(fake).mean()
                self.opt_G.zero_grad()
                g_loss.backward()
                self.opt_G.step()
                g_loss_epoch.append(g_loss.item())

            self.g_losses.append(np.mean(g_loss_epoch))
            self.c_losses.append(np.mean(c_loss_epoch))
            self.w_distances.append(np.mean(wd_epoch))

            if verbose and epoch % verbose_every == 0:
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"W-dist: {self.w_distances[-1]:.4f} | "
                    f"G: {self.g_losses[-1]:.4f} | "
                    f"C: {self.c_losses[-1]:.4f}"
                )

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """
        Computes gradient penalty: E[||∇D(x̂)||_2 - 1]²
        x̂ = epsilon * real + (1 - epsilon) * fake
        """
        bs = real.size(0)
        eps = torch.rand(bs, 1, device=self.device).expand_as(real)
        interpolated = (eps * real + (1 - eps) * fake).requires_grad_(True)
        c_interp = self.C(interpolated)

        grads = torch.autograd.grad(
            outputs=c_interp,
            inputs=interpolated,
            grad_outputs=torch.ones_like(c_interp),
            create_graph=True,
            retain_graph=True,
        )[0]
        gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    # ── generation ──────────────────────────────
    def generate(self, n: int) -> np.ndarray:
        """Generate n synthetic samples. Returns numpy array shape [n, input_dim]."""
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=self.device)
            samples = self.G(z).cpu().numpy()
        self.G.train()
        return samples

    def save(self, path: str):
        torch.save({"G": self.G.state_dict(), "C": self.C.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.C.load_state_dict(ckpt["C"])


# ─────────────────────────────────────────────
#  Multi-class WGAN-GP Manager
# ─────────────────────────────────────────────
class MultiClassWGAN:
    """
    Trains one WGAN-GP per class and manages synthetic data generation.

    Usage:
        mc = MultiClassWGAN(input_dim=122, device='cuda')
        mc.fit_all(X_train, y_train, target_per_class=10000)
        X_aug, y_aug = mc.augment(X_train, y_train)
    """

    def __init__(self, input_dim: int, device: str = "cpu", **kwargs):
        self.input_dim = input_dim
        self.device = device
        self.kwargs = kwargs
        self.trainers = {}     # class_name -> WGANGPTrainer

    def fit_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes_to_augment=None,
        min_class_weight: float = 0.10,
        epochs: int = 300,
        batch_size: int = 256,
    ):
        """
        Train a GAN for each class with weight < min_class_weight.
        If classes_to_augment is given, only train those classes.
        """
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        weights = dict(zip(unique, counts / total))

        for cls in unique:
            if classes_to_augment is not None and cls not in classes_to_augment:
                continue
            if weights[cls] >= min_class_weight and classes_to_augment is None:
                print(f"[WGAN] Skipping class '{cls}' (weight={weights[cls]:.2%})")
                continue

            X_cls = X[y == cls]
            print(f"\n[WGAN] Training for class '{cls}' | n={len(X_cls)} | weight={weights[cls]:.2%}")
            trainer = WGANGPTrainer(self.input_dim, device=self.device, **self.kwargs)
            trainer.fit(X_cls, epochs=epochs, batch_size=min(batch_size, len(X_cls)))
            self.trainers[cls] = trainer

    def augment(
        self, X: np.ndarray, y: np.ndarray, target_per_class: int = 10000
    ):
        """Augment dataset by generating synthetic samples for trained classes."""
        X_list, y_list = [X], [y]
        for cls, trainer in self.trainers.items():
            current = np.sum(y == cls)
            n_gen = max(0, target_per_class - current)
            if n_gen == 0:
                continue
            synth = trainer.generate(n_gen)
            X_list.append(synth)
            y_list.append(np.full(n_gen, cls))
            print(f"[WGAN] Generated {n_gen} samples for class '{cls}'")

        X_aug = np.vstack(X_list)
        y_aug = np.concatenate(y_list)

        # shuffle
        idx = np.random.permutation(len(X_aug))
        return X_aug[idx], y_aug[idx]
