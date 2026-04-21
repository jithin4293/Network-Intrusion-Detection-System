"""
train.py
--------
Full training pipeline for Enhanced NIDS.
Orchestrates: Preprocessing → WGAN-GP → Autoencoder → Transformer → Evaluation

Run:
    python train.py --dataset nsl-kdd --model transformer --epochs 300 --device cpu

Supported datasets : nsl-kdd | unsw-nb15 | cicids2017 | iot-23
Supported models   : transformer | dnn | cnn | all
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.preprocessing import NIDSPreprocessor
from models.wgan_gp import MultiClassWGAN
from models.autoencoder import AutoencoderTrainer
from models.transformer_classifier import (
    TransformerClassifier, DNNClassifier, CNNClassifier, ClassifierTrainer
)


# ─────────────────────────────────────────────
#  Helper: load dataset
# ─────────────────────────────────────────────
def load_dataset(dataset: str, data_dir: str = "data"):
    """
    Returns (df_train, df_test) for the chosen benchmark dataset.
    For NSL-KDD, load KDDTrain+.txt and KDDTest+.txt.
    For others, load CSV files from data_dir.
    """
    dataset = dataset.lower()
    if dataset == "nsl-kdd":
        train_path = os.path.join(data_dir, "KDDTrain+.txt")
        test_path  = os.path.join(data_dir, "KDDTest+.txt")
        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"Download NSL-KDD from https://www.unb.ca/cic/datasets/nsl.html\n"
                f"Place KDDTrain+.txt and KDDTest+.txt in '{data_dir}/'"
            )
        df_train = pd.read_csv(train_path, header=None)
        df_test  = pd.read_csv(test_path,  header=None)
    elif dataset == "unsw-nb15":
        df_train = pd.read_csv(os.path.join(data_dir, "UNSW_NB15_training-set.csv"))
        df_test  = pd.read_csv(os.path.join(data_dir, "UNSW_NB15_testing-set.csv"))
    elif dataset == "cicids2017":
        # CICIDS2017 comes as multiple CSVs; concatenate and split
        files = [f for f in os.listdir(data_dir) if "CICIDS2017" in f and f.endswith(".csv")]
        df = pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in files])
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
    elif dataset == "iot-23":
        df = pd.read_csv(os.path.join(data_dir, "CTU-IoT-Malware-Capture-34-1.csv"))
        df_train, df_test = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return df_train, df_test


# ─────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────
def run_pipeline(args):
    print("=" * 60)
    print(" Enhanced NIDS — Training Pipeline")
    print(f" Dataset : {args.dataset}")
    print(f" Model   : {args.model}")
    print(f" Device  : {args.device}")
    print("=" * 60)

    os.makedirs("outputs", exist_ok=True)

    # ── 1. Load data ───────────────────────────
    print("\n[1/5] Loading dataset...")
    df_train, df_test = load_dataset(args.dataset, args.data_dir)
    print(f"      Train: {len(df_train):,} rows | Test: {len(df_test):,} rows")

    # ── 2. Preprocessing ──────────────────────
    print("\n[2/5] Preprocessing (outlier removal, encoding, scaling)...")
    prep = NIDSPreprocessor(dataset=args.dataset)
    X_train, y_train = prep.fit_transform(df_train)
    X_test,  y_test  = prep.transform(df_test)
    class_names = prep.get_class_names()
    feature_names = prep.get_feature_names()
    input_dim = X_train.shape[1]
    n_classes = len(class_names)

    print(f"      Features : {input_dim}")
    print(f"      Classes  : {n_classes} — {class_names}")
    unique, counts = np.unique(y_train, return_counts=True)
    print("      Class distribution (train):")
    for cls, cnt in zip(unique, counts):
        print(f"        [{class_names[cls]:10s}] {cnt:6,} ({cnt/len(y_train):.1%})")

    # ── 3. WGAN-GP augmentation ────────────────
    print("\n[3/5] Training WGAN-GP for minority class augmentation...")
    mc_wgan = MultiClassWGAN(
        input_dim=input_dim,
        device=args.device,
        latent_dim=64,
        hidden_dim=256,
        n_critic=5,
        lambda_gp=10.0,
        lr=1e-4,
    )
    mc_wgan.fit_all(
        X_train, y_train,
        min_class_weight=args.gan_threshold,
        epochs=args.gan_epochs,
        batch_size=256,
    )
    X_aug, y_aug = mc_wgan.augment(X_train, y_train, target_per_class=args.target_per_class)
    print(f"      Augmented training set: {len(X_aug):,} samples")

    # ── 4. Autoencoder ─────────────────────────
    print("\n[4/5] Training autoencoder (feature extractor)...")
    ae_trainer = AutoencoderTrainer(
        input_dim=input_dim, hidden_dim=80, latent_dim=50,
        lr=1e-3, device=args.device
    )
    ae_trainer.fit(X_aug, epochs=300, batch_size=256, target_acc=0.97, patience=35)
    encoder = ae_trainer.get_encoder()
    ae_trainer.save("outputs/autoencoder.pt")
    print("      Autoencoder saved to outputs/autoencoder.pt")

    # Encode for CNN/DNN (encoder output as input)
    # For Transformer, raw features are used (transformer handles feature interactions)
    latent_dim = 50

    # ── 5. Classifier training ─────────────────
    print(f"\n[5/5] Training classifier: {args.model}...")
    results = {}

    def train_and_eval(name, model, use_encoder=True):
        print(f"\n  → {name}")
        trainer = ClassifierTrainer(
            model=model,
            n_classes=n_classes,
            lr=1e-3,
            device=args.device,
            encoder=encoder if use_encoder else None,
        )
        trainer.fit(X_aug, y_aug, epochs=args.epochs, batch_size=256,
                    X_val=X_test, y_val=y_test, patience=35, verbose=True)
        metrics = trainer.evaluate(X_test, y_test, verbose=True)
        trainer.save(f"outputs/{name.lower().replace(' ', '_')}.pt")
        results[name] = metrics
        return trainer

    if args.model in ("transformer", "all"):
        # Transformer uses raw (augmented) features — no encoder prefix
        model = TransformerClassifier(input_dim, n_classes, d_model=64, n_heads=4, n_layers=2)
        train_and_eval("Transformer", model, use_encoder=False)

    if args.model in ("dnn", "all"):
        model = DNNClassifier(latent_dim, n_classes)
        train_and_eval("G-DNNAE", model, use_encoder=True)

    if args.model in ("cnn", "all"):
        model = CNNClassifier(latent_dim, n_classes)
        train_and_eval("G-CNNAE", model, use_encoder=True)

    # ── Summary ────────────────────────────────
    print("\n" + "=" * 60)
    print(" Final Results")
    print("=" * 60)
    for name, m in results.items():
        print(f"  {name:20s} | Accuracy: {m['accuracy']:.4f} | F1: {m['f1_weighted']:.4f}")

    # Save feature names for SHAP
    np.save("outputs/feature_names.npy", np.array(feature_names))
    np.save("outputs/class_names.npy", np.array(class_names))
    print("\nOutputs saved to outputs/")
    return results


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced NIDS Training Pipeline")
    parser.add_argument("--dataset", default="nsl-kdd",
                        choices=["nsl-kdd", "unsw-nb15", "cicids2017", "iot-23"])
    parser.add_argument("--model", default="all",
                        choices=["transformer", "dnn", "cnn", "all"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--gan-epochs", type=int, default=300)
    parser.add_argument("--gan-threshold", type=float, default=0.10,
                        help="Augment classes with weight below this threshold")
    parser.add_argument("--target-per-class", type=int, default=10000,
                        help="Target number of samples per minority class after augmentation")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    run_pipeline(args)
