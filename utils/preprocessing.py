"""
preprocessing.py
----------------
Data preprocessing for Enhanced NIDS:
  - Outlier removal (MAD)
  - One-hot encoding
  - Min-max normalization
Supports: NSL-KDD, UNSW-NB15, CICIDS2017, IoT-23
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


# ─────────────────────────────────────────────
#  NSL-KDD column definitions
# ─────────────────────────────────────────────
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

NOMINAL_FEATURES_NSL = ["protocol_type", "service", "flag"]

ATTACK_MAP_NSL = {
    "normal": "Normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "httptunnel": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}


class NIDSPreprocessor:
    """
    Full preprocessing pipeline for NIDS datasets.

    Usage:
        prep = NIDSPreprocessor(dataset='nsl-kdd')
        X_train, y_train = prep.fit_transform(df_train)
        X_test,  y_test  = prep.transform(df_test)
    """

    def __init__(self, dataset: str = "nsl-kdd", mad_threshold: float = 10.0):
        self.dataset = dataset.lower()
        self.mad_threshold = mad_threshold
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.categorical_cols = []
        self.numeric_cols = []
        self.feature_names_ = []
        self._fitted = False

    # ── public API ──────────────────────────────
    def fit_transform(self, df: pd.DataFrame):
        df = df.copy()
        df, y = self._extract_labels(df)
        df = self._remove_outliers(df)
        df = self._one_hot_encode(df, fit=True)
        X = self._scale(df, fit=True)
        self.feature_names_ = list(df.columns)
        self._fitted = True
        y_enc = self.label_encoder.fit_transform(y)
        return X, y_enc

    def transform(self, df: pd.DataFrame):
        assert self._fitted, "Call fit_transform first."
        df = df.copy()
        df, y = self._extract_labels(df)
        df = self._one_hot_encode(df, fit=False)
        df = self._align_columns(df)
        X = self._scale(df, fit=False)
        y_enc = self.label_encoder.transform(y)
        return X, y_enc

    # ── internals ────────────────────────────────
    def _extract_labels(self, df):
        if self.dataset == "nsl-kdd":
            if df.shape[1] == len(NSL_KDD_COLUMNS):
                df.columns = NSL_KDD_COLUMNS
            if "difficulty" in df.columns:
                df = df.drop(columns=["difficulty"])
            y = df["label"].map(ATTACK_MAP_NSL).fillna("Other")
            df = df.drop(columns=["label"])
        elif self.dataset == "unsw-nb15":
            y = df["attack_cat"].fillna("Normal")
            df = df.drop(columns=["id", "label", "attack_cat"], errors="ignore")
        elif self.dataset == "cicids2017":
            y = df["Label"]
            df = df.drop(columns=["Label"], errors="ignore")
            df.columns = df.columns.str.strip()
        elif self.dataset == "iot-23":
            y = df["label"]
            df = df.drop(columns=["label", "uid", "id.orig_h", "id.resp_h"], errors="ignore")
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        return df, y

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """MAD-based outlier removal per numeric column, independently per class."""
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_cols = numeric
        mask = pd.Series([True] * len(df), index=df.index)
        for col in numeric:
            median = df[col].median()
            mad = (df[col] - median).abs().median()
            if mad == 0:
                continue
            sigma_hat = 1.4826 * mad
            outlier_mask = (df[col] - median).abs() > self.mad_threshold * sigma_hat
            mask = mask & ~outlier_mask
        return df[mask].reset_index(drop=True)

    def _one_hot_encode(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if self.dataset == "nsl-kdd":
            cats = [c for c in NOMINAL_FEATURES_NSL if c in df.columns]
        else:
            cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.categorical_cols = cats
        df = pd.get_dummies(df, columns=cats, drop_first=False)
        return df

    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure test set columns match training set."""
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0
        return df[self.feature_names_]

    def _scale(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        if fit:
            return self.scaler.fit_transform(df.values.astype(np.float32))
        return self.scaler.transform(df.values.astype(np.float32))

    def get_class_names(self):
        return list(self.label_encoder.classes_)

    def get_feature_names(self):
        return self.feature_names_
