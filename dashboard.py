"""
dashboard.py
------------
Streamlit real-time NIDS dashboard.
Visualises live threat detection, model metrics, and SHAP explanations.

Run:
    streamlit run dashboard.py
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enhanced NIDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 16px 20px;
        color: white;
        border: 1px solid #334155;
    }
    .metric-val { font-size: 28px; font-weight: 600; margin: 4px 0; }
    .metric-lbl { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-sub { font-size: 12px; color: #22c55e; }
    .attack-badge {
        background: #fef2f2;
        color: #dc2626;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 600;
    }
    .normal-badge {
        background: #f0fdf4;
        color: #16a34a;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ Enhanced NIDS")
    st.caption("WGAN-GP + Transformer + SHAP")
    st.divider()

    dataset = st.selectbox("Dataset", ["NSL-KDD", "UNSW-NB15", "CICIDS2017", "IoT-23"])
    model_choice = st.selectbox("Classifier", ["Transformer (Proposed)", "G-CNNAE", "G-DNNAE", "G-LSTM"])
    st.divider()
    st.subheader("Simulation Settings")
    refresh_rate = st.slider("Refresh rate (s)", 1, 10, 3)
    n_live = st.slider("Live feed entries", 5, 20, 10)
    st.divider()
    run_live = st.toggle("Enable live simulation", value=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🛡️ Enhanced AI-Based Network Intrusion Detection")
st.caption(f"Dataset: {dataset} | Model: {model_choice} | Status: {'🟢 Active' if run_live else '🔴 Paused'}")
st.divider()

# ─── Simulated metrics (replace with real model outputs in production) ────────
METRICS = {
    "Transformer (Proposed)": {"acc": 0.964, "f1": 0.951, "fp": 0.018, "tp": 0.962},
    "G-CNNAE":                {"acc": 0.932, "f1": 0.921, "fp": 0.031, "tp": 0.935},
    "G-DNNAE":                {"acc": 0.927, "f1": 0.915, "fp": 0.033, "tp": 0.928},
    "G-LSTM":                 {"acc": 0.873, "f1": 0.856, "fp": 0.042, "tp": 0.879},
}

ATTACK_TYPES = ["DDoS", "DoS", "Probe", "R2L", "U2R", "C&C", "PortScan"]
ATTACK_COLORS = {
    "DDoS": "#ef4444", "DoS": "#f97316", "Probe": "#f59e0b",
    "R2L": "#8b5cf6", "U2R": "#ec4899", "C&C": "#06b6d4", "PortScan": "#84cc16"
}

m = METRICS[model_choice]

# ─── Top metrics row ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Overall Accuracy", f"{m['acc']:.1%}", f"+{(m['acc']-0.932)*100:.1f}% vs base")
with c2:
    st.metric("Weighted F1-Score", f"{m['f1']:.3f}")
with c3:
    st.metric("False Positive Rate", f"{m['fp']:.1%}",
              delta=f"{(m['fp']-0.031)*100:.1f}%", delta_color="inverse")
with c4:
    st.metric("True Positive Rate", f"{m['tp']:.1%}")

st.divider()

# ─── Main layout ──────────────────────────────────────────────────────────────
left, right = st.columns([2, 1])

# ── Left: Live feed ──────────────────────────────────────────────────────────
with left:
    st.subheader("📡 Live Threat Feed")
    feed_placeholder = st.empty()
    chart_placeholder = st.empty()

# ── Right: Attack distribution ───────────────────────────────────────────────
with right:
    st.subheader("📊 Attack Distribution")
    dist_placeholder = st.empty()

st.divider()

# ── SHAP section ─────────────────────────────────────────────────────────────
st.subheader("🔍 SHAP Explainability")
shap_col1, shap_col2 = st.columns([1, 1])

with shap_col1:
    st.caption("Select a prediction to explain:")
    selected_attack = st.selectbox("Attack type", ATTACK_TYPES)
    explain_btn = st.button("Generate SHAP explanation ↗")

with shap_col2:
    shap_placeholder = st.empty()

st.divider()

# ── Model comparison ─────────────────────────────────────────────────────────
st.subheader("📈 Model Comparison (NSL-KDD)")
comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    st.caption("Accuracy by model")
    models = list(METRICS.keys())
    accs = [METRICS[m_]["acc"] for m_ in models]
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#0ea5e9" if m_ == model_choice else "#334155" for m_ in models]
    bars = ax.barh(models, accs, color=colors, height=0.5)
    ax.set_xlim(0.8, 1.0)
    ax.set_xlabel("Accuracy")
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    for bar, val in zip(bars, accs):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.1%}", va="center", color="white", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with comp_col2:
    st.caption("F1-score for minor attack classes")
    minor_classes = ["R2L", "U2R"]
    # Approximate per-class F1 from paper + proposed improvements
    class_f1 = {
        "Transformer (Proposed)": [0.882, 0.347],
        "G-CNNAE":                [0.800, 0.201],
        "G-DNNAE":                [0.801, 0.215],
        "G-LSTM":                 [0.652, 0.181],
    }
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(len(minor_classes))
    width = 0.2
    for i, (name, f1s) in enumerate(class_f1.items()):
        color = "#0ea5e9" if name == model_choice else "#334155"
        ax.bar(x + i * width, f1s, width, label=name, color=color)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(minor_classes, color="white")
    ax.set_ylabel("F1-score")
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─── SHAP plot generation ────────────────────────────────────────────────────
SHAP_DATA = {
    "DDoS":     {"features": ["src_bytes","dst_bytes","service","duration","flag","land","protocol"],
                 "values":   [0.42, 0.38, 0.31, 0.22, 0.18, -0.08, -0.05]},
    "Probe":    {"features": ["count","srv_count","dst_host_count","serror_rate","same_srv_rate","logged_in","flag"],
                 "values":   [0.35, 0.29, 0.24, 0.19, 0.14, -0.10, -0.06]},
    "R2L":      {"features": ["logged_in","count","su_attempted","num_root","dst_bytes","src_bytes","protocol"],
                 "values":   [0.35, 0.28, 0.24, 0.19, -0.14, -0.09, 0.07]},
    "U2R":      {"features": ["num_root","su_attempted","num_file_creations","logged_in","count","duration","protocol"],
                 "values":   [0.29, 0.25, 0.20, 0.15, -0.12, -0.07, 0.05]},
    "DoS":      {"features": ["src_bytes","duration","count","serror_rate","flag","dst_bytes","protocol"],
                 "values":   [0.44, 0.39, 0.33, 0.21, 0.16, 0.10, -0.04]},
    "C&C":      {"features": ["history","dst_bytes","src_bytes","duration","service","flag","count"],
                 "values":   [0.48, 0.36, 0.29, 0.22, 0.17, -0.07, -0.04]},
    "PortScan": {"features": ["count","srv_count","dst_host_srv_count","serror_rate","flag","duration","land"],
                 "values":   [0.51, 0.43, 0.31, 0.20, 0.14, -0.09, -0.05]},
}

if explain_btn or True:
    sd = SHAP_DATA[selected_attack]
    features = sd["features"]
    values = sd["values"]
    colors = ["#ef4444" if v > 0 else "#8b5cf6" for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(features[::-1], values[::-1], color=colors[::-1], height=0.5)
    ax.axvline(0, color="#94a3b8", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value")
    ax.set_title(f"Feature contributions — {selected_attack} prediction",
                 fontsize=11, fontweight="bold", color="white")
    ax.set_facecolor("#1e293b")
    fig.patch.set_facecolor("#1e293b")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    red_patch = mpatches.Patch(color="#ef4444", label="Pushes → ATTACK")
    blue_patch = mpatches.Patch(color="#8b5cf6", label="Pushes → NORMAL")
    ax.legend(handles=[red_patch, blue_patch], fontsize=8,
              facecolor="#0f172a", labelcolor="white")
    plt.tight_layout()
    with shap_placeholder:
        st.pyplot(fig)
    plt.close()

# ─── Live simulation loop ────────────────────────────────────────────────────
attack_counts = {a: 0 for a in ATTACK_TYPES}
feed_data = []

if run_live:
    for tick in range(1000):
        # generate fake traffic events
        new_events = []
        for _ in range(np.random.randint(1, 4)):
            is_attack = np.random.random() < 0.15
            if is_attack:
                atype = np.random.choice(ATTACK_TYPES, p=[0.3,0.15,0.2,0.15,0.08,0.07,0.05])
                attack_counts[atype] += 1
                conf = np.random.uniform(0.75, 0.99)
                status = "🔴 Blocked" if conf > 0.85 else "🟡 Flagged"
                new_events.append({
                    "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
                    "Type": atype,
                    "Src IP": f"192.168.{np.random.randint(0,255)}.{np.random.randint(1,254)}",
                    "Model": model_choice.split()[0],
                    "Confidence": f"{conf:.1%}",
                    "Status": status,
                })
            else:
                new_events.append({
                    "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
                    "Type": "Normal",
                    "Src IP": f"10.0.{np.random.randint(0,255)}.{np.random.randint(1,254)}",
                    "Model": model_choice.split()[0],
                    "Confidence": f"{np.random.uniform(0.91,0.99):.1%}",
                    "Status": "🟢 Normal",
                })

        feed_data = new_events + feed_data
        feed_data = feed_data[:n_live]

        # Render feed
        df_feed = pd.DataFrame(feed_data)
        with feed_placeholder:
            st.dataframe(df_feed, use_container_width=True, hide_index=True)

        # Activity chart
        with chart_placeholder:
            history_len = 30
            history = [np.random.randint(0, 8) for _ in range(history_len)]
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.fill_between(range(history_len), history, alpha=0.4, color="#ef4444")
            ax.plot(history, color="#ef4444", linewidth=1.5)
            ax.set_facecolor("#0f172a")
            fig.patch.set_facecolor("#0f172a")
            ax.tick_params(colors="#94a3b8", labelsize=8)
            ax.set_ylabel("Alerts", color="#94a3b8", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#334155")
            ax.spines["bottom"].set_color("#334155")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Distribution donut
        with dist_placeholder:
            labels = [k for k, v in attack_counts.items() if v > 0]
            sizes = [attack_counts[k] for k in labels]
            if sizes:
                fig, ax = plt.subplots(figsize=(4, 4))
                wedges, texts, autotexts = ax.pie(
                    sizes, labels=labels, autopct="%1.0f%%",
                    colors=[ATTACK_COLORS[l] for l in labels],
                    wedgeprops={"linewidth": 0}, startangle=90,
                    textprops={"color": "white", "fontsize": 9}
                )
                for at in autotexts:
                    at.set_color("white")
                    at.set_fontsize(8)
                ax.set_facecolor("#0f172a")
                fig.patch.set_facecolor("#0f172a")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        time.sleep(refresh_rate)
else:
    st.info("Enable live simulation in the sidebar to start the real-time feed.")
