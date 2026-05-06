# Network Anomaly Detection — Streamlit App

**Thesis:** Detecting Anomalous Network Traffic Using Machine Learning in Enterprise Networks  
**Author:** Tanbin Ahmmed | BSc (Hons) Computer Science | De Montfort University

---

## Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your trained model files
Place all four files in the same folder as `app.py`:
- `rf_model.pkl`
- `svm_model.pkl`
- `iforest_model.pkl`
- `scaler.pkl`

Generate these from your Kaggle notebook:
```python
import joblib
joblib.dump(rf_model,      "rf_model.pkl")
joblib.dump(svm_model,     "svm_model.pkl")
joblib.dump(iforest_model, "iforest_model.pkl")
joblib.dump(scaler,        "scaler.pkl")
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## App Structure

### Page 1 — Research Dashboard
- Full model comparison table (all 6 metrics)
- FPR, ROC-AUC, F1, Precision/Recall charts
- Key findings summary per model

### Page 2 — Network Analyser
- Upload any CICIDS2017-format CSV
- Choose model: Random Forest / SVM / Isolation Forest
- Network risk assessment (Low / Moderate / High)
- Per-flow prediction table (first 500 rows preview)
- Download full predictions as CSV
- Optional: if CSV has Label column, shows accuracy vs ground truth

---

## File Structure
```
streamlit_app/
├── app.py
├── requirements.txt
├── README.md
├── rf_model.pkl          ← add from Kaggle
├── svm_model.pkl         ← add from Kaggle
├── iforest_model.pkl     ← add from Kaggle
└── scaler.pkl            ← add from Kaggle
```
