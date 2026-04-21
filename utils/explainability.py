"""
explainability.py
-----------------
SHAP-based explainability for Enhanced NIDS.
Key novelty — base paper has NO interpretability.

Supports:
  - Per-prediction SHAP waterfall plots
  - Global feature importance (beeswarm)
  - Summary bar charts
  - Text explanation for SOC analysts

Requires: pip install shap matplotlib
"""

import numpy as np
import torch
import torch.nn as nn

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[SHAP] shap not installed. Run: pip install shap")

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


class NIDSExplainer:
    """
    SHAP explainer wrapper for NIDS classifiers.

    Supports DeepExplainer (neural nets) and KernelExplainer (model-agnostic).

    Usage:
        explainer = NIDSExplainer(trainer, feature_names, class_names)
        explainer.fit_background(X_train[:200])
        shap_vals = explainer.explain(X_sample)
        explainer.plot_waterfall(shap_vals, idx=0)
        explainer.plot_summary(shap_vals)
        text = explainer.text_explanation(shap_vals, idx=0, pred_class='DDoS')
    """

    def __init__(self, trainer, feature_names: list, class_names: list, device: str = "cpu"):
        assert SHAP_AVAILABLE, "Install shap: pip install shap"
        self.trainer = trainer
        self.feature_names = feature_names
        self.class_names = class_names
        self.device = torch.device(device)
        self._shap_explainer = None
        self._background = None

    # ── setup ──────────────────────────────────
    def fit_background(self, X_background: np.ndarray, n_background: int = 100):
        """
        Fit SHAP explainer on background samples.
        Uses DeepExplainer for PyTorch models (fast, exact for linear layers).
        Falls back to KernelExplainer for sklearn models.
        """
        bg = X_background[:n_background]
        self._background = bg

        model = self.trainer.model
        encoder = self.trainer.encoder

        # Build a wrapper that applies encoder + classifier
        class FullModel(nn.Module):
            def __init__(self, enc, clf, dev):
                super().__init__()
                self.enc = enc
                self.clf = clf
                self.dev = dev

            def forward(self, x):
                if self.enc is not None:
                    with torch.no_grad():
                        x = self.enc(x)
                return torch.softmax(self.clf(x), dim=1)

        full_model = FullModel(encoder, model, self.device).to(self.device)
        full_model.eval()

        bg_t = torch.FloatTensor(bg).to(self.device)
        self._shap_explainer = shap.DeepExplainer(full_model, bg_t)
        print(f"[SHAP] Explainer fitted on {len(bg)} background samples.")

    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for X.
        Returns array of shape [n_classes, n_samples, n_features].
        """
        assert self._shap_explainer is not None, "Call fit_background first."
        X_t = torch.FloatTensor(X).to(self.device)
        shap_values = self._shap_explainer.shap_values(X_t)
        return shap_values

    # ── plots ───────────────────────────────────
    def plot_waterfall(self, shap_values, idx: int = 0, class_idx: int = None,
                       pred_label: str = None, save_path: str = None):
        """
        Waterfall plot for a single prediction — shows how each feature
        contributes to pushing the prediction from baseline to final value.
        """
        assert MPL_AVAILABLE, "Install matplotlib."
        if isinstance(shap_values, list):
            cls = class_idx if class_idx is not None else np.argmax(
                [abs(sv[idx]).sum() for sv in shap_values]
            )
            sv = shap_values[cls][idx]
            title = f"SHAP — {pred_label or self.class_names[cls]}"
        else:
            sv = shap_values[idx]
            title = f"SHAP — Sample {idx}"

        # Sort by absolute value
        order = np.argsort(np.abs(sv))[::-1][:20]
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = ["#ef4444" if v > 0 else "#8b5cf6" for v in sv[order]]
        ax.barh(range(len(order)), sv[order], color=colors)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([self.feature_names[i] if i < len(self.feature_names)
                            else f"f{i}" for i in order], fontsize=9)
        ax.axvline(0, color="#334155", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP value (contribution to prediction)")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.invert_yaxis()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[SHAP] Saved waterfall plot to {save_path}")
        plt.show()

    def plot_summary(self, shap_values, class_idx: int = 0,
                     X: np.ndarray = None, save_path: str = None):
        """
        Beeswarm / summary bar chart of global feature importance.
        """
        assert MPL_AVAILABLE, "Install matplotlib."
        sv = shap_values[class_idx] if isinstance(shap_values, list) else shap_values
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:20]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(range(len(order)), mean_abs[order], color="#0ea5e9")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([self.feature_names[i] if i < len(self.feature_names)
                            else f"f{i}" for i in order], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Global Feature Importance — {self.class_names[class_idx]}",
                     fontsize=12, fontweight="bold", pad=12)
        ax.invert_yaxis()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    # ── text explanation ─────────────────────────
    def text_explanation(self, shap_values, idx: int, pred_class: str,
                         confidence: float, top_k: int = 5) -> str:
        """
        Generate human-readable explanation for SOC analysts.
        Example output:
          'The model classified this flow as DDoS (confidence: 98.2%).
           Top contributing features:
             + src_bytes     (+0.42) — unusually high outbound bytes
             + dst_bytes     (+0.38) — high destination traffic
             + service       (+0.31) — suspicious service type
             - land          (-0.08) — normal (pushes toward Normal)'
        """
        if isinstance(shap_values, list):
            cls_idx = self.class_names.index(pred_class) if pred_class in self.class_names else 0
            sv = shap_values[cls_idx][idx]
        else:
            sv = shap_values[idx]

        order = np.argsort(np.abs(sv))[::-1][:top_k]
        lines = [
            f"Prediction: {pred_class} (confidence: {confidence:.1%})",
            f"Top {top_k} contributing features:",
        ]
        for rank, i in enumerate(order, 1):
            fname = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            val = sv[i]
            direction = "pushes toward ATTACK" if val > 0 else "pushes toward NORMAL"
            lines.append(f"  {rank}. {fname:30s} SHAP={val:+.3f}  ({direction})")

        return "\n".join(lines)

    # ── sklearn fallback ─────────────────────────
    def fit_kernel_explainer(self, predict_fn, X_background: np.ndarray,
                             n_background: int = 50):
        """
        Kernel SHAP — slower but works for any model (sklearn, etc.).
        Use for SVM/DT comparison models.
        """
        bg = shap.sample(X_background, n_background)
        self._shap_explainer = shap.KernelExplainer(predict_fn, bg)
        self._background = bg
        print(f"[SHAP] KernelExplainer fitted on {len(bg)} background samples.")

    def kernel_explain(self, X: np.ndarray, n_samples: int = 100) -> np.ndarray:
        return self._shap_explainer.shap_values(X[:n_samples])
