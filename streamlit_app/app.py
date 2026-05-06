# ============================================================
# Network Anomaly Detection — Streamlit App
# Thesis: Detecting Anomalous Network Traffic Using ML
# Author: Tanbin Ahmmed | De Montfort University
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Network Anomaly Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .risk-low    { background:#d4edda; border-left:5px solid #28a745;
                   padding:12px 16px; border-radius:6px; margin:8px 0; }
    .risk-medium { background:#fff3cd; border-left:5px solid #ffc107;
                   padding:12px 16px; border-radius:6px; margin:8px 0; }
    .risk-high   { background:#f8d7da; border-left:5px solid #dc3545;
                   padding:12px 16px; border-radius:6px; margin:8px 0; }
    .model-card  { background:#f8f9fa; border:1px solid #dee2e6;
                   border-radius:10px; padding:16px; margin-bottom:12px; }
    .criteria-ok   { color:#28a745; font-weight:bold; }
    .criteria-fail { color:#dc3545; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ── The 77 required feature columns ──────────────────────────
REQUIRED_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s",
    "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

# ── Columns dropped during preprocessing (not required) ──────
DROP_COLS = [
    "Flow ID", "Source IP", "Destination IP",
    "Source Port", "Destination Port", "Timestamp"
]

# ── Thesis results ────────────────────────────────────────────
THESIS_RESULTS = {
    "Random Forest": {
        "Accuracy": 0.9986, "Precision": 0.9954, "Recall": 0.9965,
        "F1-Score": 0.9960, "ROC-AUC":   0.9999, "FPR":    0.0009,
    },
    "SVM": {
        "Accuracy": 0.9567, "Precision": 0.8208, "Recall": 0.9510,
        "F1-Score": 0.8811, "ROC-AUC":   0.9913, "FPR":    0.0422,
    },
    "Isolation Forest": {
        "Accuracy": 0.7925, "Precision": 0.4208, "Recall": 0.6076,
        "F1-Score": 0.4972, "ROC-AUC":   0.8106, "FPR":    0.1700,
    },
}

MODEL_COLORS = {
    "Random Forest":    "#2196F3",
    "SVM":              "#FF9800",
    "Isolation Forest": "#4CAF50",
}


# ── Cached model loader ───────────────────────────────────────
@st.cache_resource
def load_models():
    models, missing = {}, []
    for name, fname in [
        ("Random Forest",    "rf_model.pkl"),
        ("SVM",              "svm_model.pkl"),
        ("Isolation Forest", "iforest_model.pkl"),
    ]:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            missing.append(fname)
    scaler_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "scaler.pkl"
    )
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return models, scaler, missing


# ── Template CSV generator ────────────────────────────────────
@st.cache_data
def generate_template_csv():
    """
    Returns a CSV bytes object with the 77 required column headers
    and one example row of placeholder values.
    """
    example_row = {col: 0.0 for col in REQUIRED_FEATURES}
    example_row["Flow Duration"]          = 1000000
    example_row["Total Fwd Packets"]      = 10
    example_row["Total Backward Packets"] = 8
    example_row["Flow Bytes/s"]           = 1500.0
    example_row["Flow Packets/s"]         = 18.0
    df_template = pd.DataFrame([example_row])
    buf = io.BytesIO()
    df_template.to_csv(buf, index=False)
    return buf.getvalue()


# ── File validation function ──────────────────────────────────
def validate_upload(df: pd.DataFrame, training_features: list):
    """
    Validates an uploaded DataFrame against the required features.
    Returns (is_valid, missing_cols, non_numeric_cols, extra_cols, warnings)
    """
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Drop known non-feature columns silently
    drop_present = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_present)

    # Drop Label column if present
    has_labels = "Label" in df.columns
    if has_labels:
        df = df.drop(columns=["Label"])

    uploaded_cols    = set(df.columns.tolist())
    required_cols    = set(training_features)

    missing_cols     = sorted(required_cols - uploaded_cols)
    extra_cols       = sorted(uploaded_cols - required_cols)

    # Check for non-numeric values in required columns
    non_numeric_cols = []
    for col in training_features:
        if col in df.columns:
            try:
                pd.to_numeric(df[col], errors="raise")
            except (ValueError, TypeError):
                non_numeric_cols.append(col)

    is_valid = (len(missing_cols) == 0 and len(non_numeric_cols) == 0)
    warnings = []
    if extra_cols:
        warnings.append(
            f"Your file has {len(extra_cols)} extra column(s) not used by "
            f"the model. They will be ignored: "
            f"`{'`, `'.join(extra_cols[:5])}`"
            + (" ..." if len(extra_cols) > 5 else "")
        )

    return is_valid, missing_cols, non_numeric_cols, extra_cols, warnings, df, has_labels


# ── Highlight best in results table ──────────────────────────
def highlight_best(s):
    best = s.min() if s.name == "FPR" else s.max()
    return ["background-color:#d4edda; font-weight:bold"
            if v == best else "" for v in s]


# ── Sidebar ───────────────────────────────────────────────────
try:
    st.sidebar.image(
        "C:\\Users\\tanbi\\ids_thesis\\notebooks\\Network anomaly detection.png",
        width=210, caption="Detect your network now! Upload a CSV with the 77 required features and see how all three models perform side by side."
    )
except Exception:
    pass

st.sidebar.markdown("---")
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠  Research Dashboard", "🔍  Network Analyser"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Thesis**  
Detecting Anomalous Network Traffic Using Machine Learning in Enterprise Networks

**Author**  
Tanbin Ahmmed  
B.Sc. (Hons) in Computer Science  
De Montfort University
""")


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — RESEARCH DASHBOARD
# ═══════════════════════════════════════════════════════════════
if page == "🏠  Research Dashboard":

    st.title("🛡️ Network Anomaly Detection — Research Dashboard")
    st.markdown(
        "Experimental results from a BSc thesis investigating machine learning-based "
        "intrusion detection for enterprise networks, with primary emphasis on "
        "**false-positive rate reduction**."
    )
    st.markdown("---")

    # Dataset overview
    st.markdown("### 📊 Dataset Overview — CICIDS2017")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Samples",    "2,520,798")
    c2.metric("Features",         "77")
    c3.metric("Training Samples", "2,016,638")
    c4.metric("Test Samples",     "504,160")
    c5.metric("Imbalance Ratio",  "4.9 : 1")
    st.caption(
        "All 8 daily CICIDS2017 capture files used · Binary classification: "
        "BENIGN = 0, ATTACK = 1 · 83.1% benign / 16.9% attack"
    )
    st.markdown("---")

    # Results table
    st.markdown("### 📋 Model Comparison — All Metrics")
    df_r = pd.DataFrame(THESIS_RESULTS).T.reset_index()
    df_r.columns = ["Model", "Accuracy", "Precision",
                    "Recall", "F1-Score", "ROC-AUC", "FPR"]
    st.dataframe(
        df_r.set_index("Model").style
        .apply(highlight_best)
        .format("{:.4f}"),
        use_container_width=True
    )
    st.caption(
        "Green = best value per metric. "
        "FPR: lower is better. All other metrics: higher is better."
    )
    st.markdown("---")

    # Charts
    st.markdown("### 📈 Visual Comparison")
    model_names = list(THESIS_RESULTS.keys())
    colors      = [MODEL_COLORS[m] for m in model_names]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Model Performance Comparison — CICIDS2017",
                 fontsize=13, fontweight="bold")

    def make_bar(ax, values, title, ylabel, ylim=None):
        bars = ax.bar(model_names, values, color=colors,
                      edgecolor="white", width=0.5)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(*(ylim if ylim else (0, max(values) * 1.35)))
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.02,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)

    make_bar(axes[0, 0],
             [THESIS_RESULTS[m]["FPR"]      for m in model_names],
             "False-Positive Rate (lower is better)", "FPR")
    make_bar(axes[0, 1],
             [THESIS_RESULTS[m]["ROC-AUC"]  for m in model_names],
             "ROC-AUC (higher is better)", "AUC", ylim=(0.7, 1.05))
    make_bar(axes[1, 0],
             [THESIS_RESULTS[m]["F1-Score"] for m in model_names],
             "F1-Score (higher is better)", "F1", ylim=(0.4, 1.05))

    x, w = np.arange(len(model_names)), 0.35
    precs = [THESIS_RESULTS[m]["Precision"] for m in model_names]
    recs  = [THESIS_RESULTS[m]["Recall"]    for m in model_names]
    axes[1, 1].bar(x - w/2, precs, w, label="Precision",
                   color="#2196F3", alpha=0.85)
    axes[1, 1].bar(x + w/2, recs,  w, label="Recall",
                   color="#FF9800", alpha=0.85)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(model_names, fontsize=9)
    axes[1, 1].set_ylim(0.3, 1.1)
    axes[1, 1].set_ylabel("Score", fontsize=9)
    axes[1, 1].set_title("Precision vs Recall", fontsize=10,
                          fontweight="bold", pad=6)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].spines["top"].set_visible(False)
    axes[1, 1].spines["right"].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Key findings
    st.markdown("### 🔑 Key Findings")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.success("**Random Forest — Best Overall**")
        st.markdown("""
- FPR **0.09%** — lowest by far
- ~377 false alarms per 504,160 flows
- Accuracy **99.86%** · F1 **0.9960**
- ✅ Recommended for enterprise deployment
        """)
    with col_b:
        st.warning("**SVM — Competitive but Constrained**")
        st.markdown("""
- FPR **4.22%** — 47× higher than RF
- ~17,682 false alarms per test set
- Strong recall **(0.9510)**
- ⚠️ Limited by 100k training sample
        """)
    with col_c:
        st.info("**Isolation Forest — Unsupervised Trade-off**")
        st.markdown("""
- FPR **17.0%** — highest false alarm rate
- No attack labels used in training
- ROC-AUC **0.81** without supervision
- ℹ️ Best for novel/zero-day detection
        """)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — NETWORK ANALYSER
# ═══════════════════════════════════════════════════════════════
else:

    st.title("🔍 Network Analyser — Multi-Model Prediction")
    st.markdown(
        "Upload your own network traffic CSV file. All three trained models analyse "
        "it simultaneously and results are compared side by side. Read the criteria "
        "below before uploading to ensure your file is compatible."
    )
    st.markdown("---")

    models, scaler, missing = load_models()

    if missing:
        st.warning(
            f"⚠️ Missing model files: `{'`, `'.join(missing)}`. "
            "Place the `.pkl` files in the same folder as `app.py`."
        )
    if not models:
        st.error("No models loaded. Cannot run predictions.")
        st.stop()
    if scaler is None:
        st.error("`scaler.pkl` not found.")
        st.stop()

    # Get training feature names from scaler
    training_features = (
        list(scaler.feature_names_in_)
        if hasattr(scaler, "feature_names_in_")
        else REQUIRED_FEATURES
    )

    # ── Section 1: Criteria ───────────────────────────────────
    st.markdown("## 📋 Upload Criteria")
    st.markdown(
        "Your CSV file must meet all of the following criteria to be accepted. "
        "Files that do not meet the criteria will receive a detailed error "
        "explaining exactly what is missing or wrong."
    )

    with st.expander("✅ Criteria Checklist — expand to read before uploading",
                     expanded=True):

        st.markdown("### Required Criteria")
        st.markdown("""
| # | Criterion | Details |
|---|---|---|
| 1 | **File format** | Must be a `.csv` file |
| 2 | **Column names** | Must contain all 77 required feature columns (exact names, case-sensitive) |
| 3 | **Data types** | All feature columns must contain numeric values only |
| 4 | **No empty file** | File must contain at least 1 data row |
| 5 | **Encoding** | File must be UTF-8 encoded |

### Optional Columns (ignored if present)
| Column | Behaviour |
|---|---|
| `Label` | If present, the app evaluates prediction accuracy against it |
| `Flow ID`, `Source IP`, `Destination IP`, `Source Port`, `Destination Port`, `Timestamp` | Automatically dropped before prediction |
| Any other extra columns | Automatically ignored — will not cause an error |

### How to Get the Required 77 Features
Your network traffic data must be processed through **CICFlowMeter** to generate
these features. CICFlowMeter takes a `.pcap` capture file and outputs a CSV with
the exact columns required.

**Steps:**
1. Capture your traffic as `.pcap` using [Wireshark](https://www.wireshark.org/)
2. Process through [CICFlowMeter](https://www.unb.ca/cic/research/applications.html)
3. Upload the resulting CSV here

Alternatively, any of the **8 CICIDS2017 daily CSV files** are ready to use without
any processing.
        """)

        st.markdown("---")
        st.markdown("### 📄 All 77 Required Column Names")
        st.markdown(
            "Your file must contain **all** of the following columns. "
            "Column names are **case-sensitive** and must match exactly."
        )

        # Display as a searchable 3-column layout
        df_cols = pd.DataFrame({
            "Feature Name": training_features,
            "Type": ["Numeric (float/int)"] * len(training_features),
            "#": list(range(1, len(training_features) + 1))
        })[["#", "Feature Name", "Type"]]

        st.dataframe(df_cols, use_container_width=True, height=300)

        st.markdown("---")

        # Template download
        st.markdown("### 📥 Download CSV Template")
        st.markdown(
            "Download this template to see the exact column structure required. "
            "It contains the 77 column headers and one example row with placeholder values. "
            "Replace the placeholder values with your own network flow data."
        )
        st.download_button(
            label="⬇️ Download CSV Template (77 columns)",
            data=generate_template_csv(),
            file_name="network_analyser_template.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption(
            "Template contains one example row with placeholder values. "
            "Replace with real CICFlowMeter-generated data for accurate predictions."
        )

    st.markdown("---")

    # ── Section 2: Upload ─────────────────────────────────────
    st.markdown("## 📁 Upload Your Network Traffic CSV")
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help=(
            "Must contain all 77 required feature columns. "
            "Download the template above to see the exact format required."
        )
    )

    if uploaded_file is not None:

        # ── Load raw file ─────────────────────────────────────
        with st.spinner("Reading file..."):
            try:
                df_raw = pd.read_csv(
                    uploaded_file, encoding="utf-8", low_memory=False
                )
            except UnicodeDecodeError:
                st.error(
                    "❌ **Encoding error** — your file is not UTF-8 encoded. "
                    "Open your file in Excel or a text editor and re-save it "
                    "as UTF-8 CSV before uploading."
                )
                st.stop()
            except Exception as e:
                st.error(f"❌ **Could not read file:** {e}")
                st.stop()

        if len(df_raw) == 0:
            st.error(
                "❌ **Empty file** — your CSV has no data rows. "
                "Please upload a file containing at least one network flow."
            )
            st.stop()

        st.success(
            f"✅ File loaded: **{uploaded_file.name}** — "
            f"{df_raw.shape[0]:,} rows · {df_raw.shape[1]} columns"
        )

        # ── Validate ──────────────────────────────────────────
        st.markdown("### 🔍 Validating Your File...")

        is_valid, missing_cols, non_numeric_cols, extra_cols, \
            val_warnings, df_clean, has_labels = validate_upload(
                df_raw.copy(), training_features
            )

        # Show warnings for extra columns
        for w in val_warnings:
            st.warning(f"⚠️ {w}")

        # Show label detection
        if has_labels:
            st.info(
                "ℹ️ `Label` column detected — after prediction the app will "
                "automatically evaluate accuracy against your ground truth labels."
            )

        # ── Validation errors ─────────────────────────────────
        if missing_cols or non_numeric_cols:
            st.error("❌ **Your file does not meet the required criteria.**")

            if missing_cols:
                st.markdown("#### Missing Columns")
                st.markdown(
                    f"Your file is missing **{len(missing_cols)}** of the "
                    f"77 required feature columns. The model cannot run without them."
                )

                # Show missing columns in a clear table
                df_missing = pd.DataFrame({
                    "Missing Column Name": missing_cols,
                    "Expected Type": ["Numeric (float/int)"] * len(missing_cols)
                })
                st.dataframe(df_missing, use_container_width=True)

                st.markdown("""
**Why is this happening?**
Your CSV was not generated by CICFlowMeter, or the column names have been
renamed or reformatted. The model was trained on CICFlowMeter output and
expects exact column names.

**How to fix it:**
- If you have a `.pcap` file → process it through
  [CICFlowMeter](https://www.unb.ca/cic/research/applications.html)
- If you renamed columns → restore the original CICFlowMeter column names
- If you are testing → download the **CSV Template** above and use it as
  a starting point
                """)

            if non_numeric_cols:
                st.markdown("#### Non-Numeric Values Detected")
                st.markdown(
                    f"**{len(non_numeric_cols)}** column(s) contain non-numeric "
                    f"values. All feature columns must contain numbers only."
                )
                df_nonnumeric = pd.DataFrame({
                    "Column with Non-Numeric Values": non_numeric_cols,
                    "Required Type": ["Numeric (float/int)"] * len(non_numeric_cols)
                })
                st.dataframe(df_nonnumeric, use_container_width=True)

                st.markdown("""
**How to fix it:**
- Check the listed columns for text values, symbols, or empty cells
- Replace any non-numeric entries with `0` or the correct numeric value
- Make sure there are no units (e.g. "1500 bytes" should just be `1500`)
                """)

            st.markdown("---")
            st.markdown(
                "📥 **Need help?** Download the CSV Template above to see the "
                "exact column structure and data format required."
            )
            st.stop()

        # ── All criteria met ──────────────────────────────────
        st.success(
            f"✅ **File passed all criteria** — "
            f"{len(training_features)} required columns found · "
            f"All values numeric · Ready for prediction"
        )

        # Final preprocessing
        with st.spinner("Preprocessing..."):
            try:
                df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
                df_clean.dropna(inplace=True)
                rows_after_clean = len(df_clean)

                if rows_after_clean < len(df_raw):
                    st.warning(
                        f"⚠️ {len(df_raw) - rows_after_clean:,} rows were removed "
                        f"during cleaning (contained infinite or missing values). "
                        f"{rows_after_clean:,} rows will be analysed."
                    )

                X_scaled = scaler.transform(df_clean[training_features])

            except Exception as e:
                st.error(f"❌ Preprocessing error: {e}")
                st.stop()

        # ── Run all three models ──────────────────────────────
        all_preds = {}
        with st.spinner("Running all three models simultaneously..."):
            for name, model in models.items():
                try:
                    if name == "Isolation Forest":
                        raw = model.predict(X_scaled)
                        all_preds[name] = (raw == -1).astype(int)
                    else:
                        all_preds[name] = model.predict(X_scaled)
                except Exception as e:
                    st.warning(f"⚠️ {name} prediction failed: {e}")

        if not all_preds:
            st.error("All models failed to produce predictions.")
            st.stop()

        total = len(X_scaled)
        st.markdown("---")

        # ── Network overview cards ────────────────────────────
        st.markdown("## 📊 Network Overview — All Three Models")
        overview_cols = st.columns(len(all_preds))

        for col, (name, preds) in zip(overview_cols, all_preds.items()):
            n_attack = int(preds.sum())
            n_benign = total - n_attack
            atk_pct  = n_attack / total * 100
            tfpr     = THESIS_RESULTS.get(name, {}).get("FPR", 0)

            with col:
                st.markdown(
                    f"<div class='model-card'>"
                    f"<b style='color:{MODEL_COLORS[name]};font-size:15px'>"
                    f"{name}</b><br><br>"
                    f"🟢 Benign: <b>{n_benign:,}</b> ({100-atk_pct:.1f}%)<br>"
                    f"🔴 Attack: <b>{n_attack:,}</b> ({atk_pct:.1f}%)<br><br>"
                    f"📌 Trained FPR: <b>{tfpr:.2%}</b>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                if atk_pct < 1:
                    st.markdown(
                        "<div class='risk-low'>🟢 Low Risk</div>",
                        unsafe_allow_html=True
                    )
                elif atk_pct < 10:
                    st.markdown(
                        "<div class='risk-medium'>🟡 Moderate Risk</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        "<div class='risk-high'>🔴 High Risk</div>",
                        unsafe_allow_html=True
                    )

        st.markdown("---")

        # ── Comparison charts ─────────────────────────────────
        st.markdown("## 📈 Prediction Comparison")

        pred_names = list(all_preds.keys())
        atk_counts = [int(all_preds[n].sum())   for n in pred_names]
        ben_counts = [total - a                  for a in atk_counts]
        atk_pcts   = [a / total * 100            for a in atk_counts]
        clrs       = [MODEL_COLORS[n]            for n in pred_names]
        x2, w2     = np.arange(len(pred_names)), 0.35

        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

        # Grouped benign/attack counts
        axes2[0].bar(x2 - w2/2, ben_counts, w2,
                     label="Benign", color="#4CAF50", alpha=0.85)
        axes2[0].bar(x2 + w2/2, atk_counts, w2,
                     label="Attack", color="#F44336", alpha=0.85)
        axes2[0].set_xticks(x2)
        axes2[0].set_xticklabels(pred_names, fontsize=9)
        axes2[0].set_title("Benign vs Attack Count per Model",
                           fontsize=11, fontweight="bold")
        axes2[0].set_ylabel("Number of flows")
        axes2[0].legend()
        axes2[0].spines["top"].set_visible(False)
        axes2[0].spines["right"].set_visible(False)

        # Attack rate %
        bars2 = axes2[1].bar(pred_names, atk_pcts, color=clrs,
                              edgecolor="white", width=0.5)
        axes2[1].set_title("Predicted Attack Rate (%)",
                           fontsize=11, fontweight="bold")
        axes2[1].set_ylabel("% of flows flagged as attack")
        axes2[1].set_ylim(0, max(atk_pcts) * 1.4 + 1)
        for bar, val in zip(bars2, atk_pcts):
            axes2[1].text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + 0.3,
                          f"{val:.2f}%", ha="center", va="bottom",
                          fontsize=10, fontweight="bold")
        axes2[1].spines["top"].set_visible(False)
        axes2[1].spines["right"].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.markdown("---")

        # ── Model agreement ───────────────────────────────────
        st.markdown("## 🤝 Model Agreement Analysis")

        pred_arrays = list(all_preds.values())
        if len(pred_arrays) >= 2:
            stacked    = np.stack(pred_arrays, axis=1)
            majority   = (stacked.sum(axis=1) >= 2).astype(int)
            mv_atk     = int(majority.sum())
            mv_ben     = total - mv_atk
            full_agree = int(
                (stacked.min(axis=1) == stacked.max(axis=1)).sum()
            )
            agree_pct  = full_agree / total * 100

            m1, m2, m3 = st.columns(3)
            m1.metric("Majority Vote — Attacks",
                      f"{mv_atk:,}", f"{mv_atk/total*100:.1f}%")
            m2.metric("Majority Vote — Benign",
                      f"{mv_ben:,}", f"{mv_ben/total*100:.1f}%")
            m3.metric("Full Model Agreement",
                      f"{agree_pct:.1f}%",
                      f"{full_agree:,} of {total:,} flows")

            st.caption(
                "Majority Vote = flagged ATTACK by ≥2 models. "
                "Full Agreement = all three models assign the same label."
            )

            with st.expander("ℹ️ How to interpret model agreement"):
                st.markdown("""
| Agreement Pattern | Interpretation | Recommended Action |
|---|---|---|
| **All 3 flag ATTACK** | Very high confidence — likely a real attack | Investigate immediately |
| **RF + SVM flag ATTACK** | High confidence — supervised models agree | Review as priority |
| **Only Isolation Forest flags ATTACK** | Possible novel/zero-day pattern | Worth reviewing |
| **Only SVM flags ATTACK** | May be a false positive (SVM has higher FPR) | Lower priority review |
| **All 3 flag BENIGN** | High confidence — legitimate traffic | No action needed |
                """)

        st.markdown("---")

        # ── Evaluation against ground truth ───────────────────
        if has_labels:
            from sklearn.metrics import (
                accuracy_score, f1_score,
                precision_score, recall_score, confusion_matrix
            )
            # Strip column names in df_raw to handle whitespace variants
            df_raw.columns = df_raw.columns.str.strip()
            label_col = "Label"
            if label_col not in df_raw.columns:
                # Fallback — find any column that looks like Label
                matches = [c for c in df_raw.columns
                           if c.strip().lower() == "label"]
                label_col = matches[0] if matches else None

            if label_col:
                true_labels = df_raw.iloc[:total][label_col].apply(
                    lambda x: 0 if str(x).strip() == "BENIGN" else 1
                ).values
            else:
                has_labels = False
                st.warning("Label column could not be located after stripping — "
                           "skipping ground truth evaluation.")

            st.markdown("## ✅ Evaluation Against Your True Labels")
            st.info(
                "Your file contained a `Label` column. "
                "The table below shows how accurately each model predicted "
                "your ground truth labels."
            )

            eval_rows = []
            for name, preds in all_preds.items():
                n = min(len(true_labels), len(preds))
                tn, fp, fn, tp = confusion_matrix(
                    true_labels[:n], preds[:n]
                ).ravel()
                eval_rows.append({
                    "Model":     name,
                    "Accuracy":  round(accuracy_score(
                        true_labels[:n], preds[:n]), 4),
                    "Precision": round(precision_score(
                        true_labels[:n], preds[:n], zero_division=0), 4),
                    "Recall":    round(recall_score(
                        true_labels[:n], preds[:n], zero_division=0), 4),
                    "F1-Score":  round(f1_score(
                        true_labels[:n], preds[:n], zero_division=0), 4),
                    "FPR":       round(
                        fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
                })

            st.dataframe(
                pd.DataFrame(eval_rows).set_index("Model")
                .style.apply(highlight_best).format("{:.4f}"),
                use_container_width=True
            )
            st.markdown("---")

        # ── Per-flow predictions table ────────────────────────
        st.markdown("## 🗂️ Per-Flow Predictions (first 500 rows)")

        df_out = df_raw.iloc[:total].copy()
        for name, preds in all_preds.items():
            col_key = f"Pred_{name.replace(' ', '_')}"
            df_out[col_key] = [
                "ATTACK" if p == 1 else "BENIGN" for p in preds
            ]
        if len(pred_arrays) >= 2:
            df_out["Majority_Vote"] = [
                "ATTACK" if p == 1 else "BENIGN" for p in majority
            ]

        pred_cols  = [c for c in df_out.columns
                      if c.startswith("Pred_") or c == "Majority_Vote"]
        other_cols = [c for c in df_out.columns if c not in pred_cols]

        st.dataframe(
            df_out[pred_cols + other_cols].head(500),
            use_container_width=True
        )
        st.markdown("---")

        # ── Downloads ─────────────────────────────────────────
        st.markdown("## 💾 Download Results")

        @st.cache_data
        def to_csv(d):
            return d.to_csv(index=False).encode("utf-8")

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                label="⬇️ Full results — all features + predictions",
                data=to_csv(df_out),
                file_name="network_predictions_full.csv",
                mime="text/csv",
                use_container_width=True
            )
        with dl2:
            st.download_button(
                label="⬇️ Predictions only — compact summary",
                data=to_csv(df_out[pred_cols]),
                file_name="network_predictions_summary.csv",
                mime="text/csv",
                use_container_width=True
            )
        st.caption(
            "Prediction columns: BENIGN = legitimate traffic · "
            "ATTACK = anomalous or malicious flow · "
            "Majority_Vote = label agreed on by ≥2 models."
        )

        # Model guidance
        with st.expander("ℹ️ Which model's predictions should I trust most?"):
            st.markdown("""
| Model | Trained FPR | Best for | Notes |
|---|---|---|---|
| **Random Forest** | 0.09% | General use, fewest false alarms | ✅ Most reliable overall |
| **SVM** | 4.22% | High recall — catches more attacks | ⚠️ More false positives |
| **Isolation Forest** | 17.0% | Novel / zero-day attacks | ℹ️ No labels needed in training |

Use **Majority_Vote** as your primary column for the most balanced result.
            """)

    else:
        # ── No file uploaded yet ──────────────────────────────
        st.markdown("## 📋 How to Use")
        st.markdown("""
1. **Read the criteria** in the section above — your file must meet all requirements
2. **Download the CSV template** if you are unsure about the format
3. **Upload your CSV** using the uploader
4. All three models run automatically — no further configuration needed
5. **Review** the side-by-side risk assessment and model agreement analysis
6. **Download** the full annotated results with one prediction column per model
        """)

        col_l, col_r = st.columns(2)
        with col_l:
            st.info(
                "💡 **Quick test:** Upload any of the 8 CICIDS2017 daily CSV files "
                "to see the analyser in action immediately."
            )
        with col_r:
            st.info(
                "💡 **Your own traffic:** Capture with Wireshark → process through "
                "CICFlowMeter → upload here. See criteria above for details."
            )
