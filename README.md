# Detecting Anomalous Network Traffic Using Machine Learning

**BSc (Hons) Computer Science — Development Project**
**Author:** Tanbin Ahmmed | De Montfort University | 2025–2026
**Supervisor:** Samuel Agbesi

---

## Project Overview

This repository contains the full implementation for the thesis:
*"Detecting Anomalous Network Traffic Using Machine Learning in Enterprise Networks"*

Three machine learning models were implemented and evaluated on the CICIDS2017 dataset with primary emphasis on false-positive rate reduction:

| Model | Accuracy | FPR |
|---|---|---|
| Random Forest | 0.9986 | 0.0009 |
| SVM | 0.9567 | 0.0422 |
| Isolation Forest | 0.7925 | 0.1700 |

---

## Repository Structure

```
network-anomaly-detection/
├── notebooks/
│   └── training.ipynb       # Full training pipeline
├── streamlit_app/
│   ├── app.py               # Network analyser application
│   ├── requirements.txt
│   ├── rf_model.pkl
│   ├── svm_model.pkl
│   ├── iforest_model.pkl
│   └── scaler.pkl
├── figures/                 # All evaluation charts
└── README.md
```

---

## Dataset

This project uses the [CICIDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html) from the Canadian Institute for Cybersecurity. Download the CSV files and place them in a `data/` folder before running the notebook.

---

## Setup and Running

### Training the models

```bash
conda create -n ids_thesis python=3.11 -y
conda activate ids_thesis
pip install pandas numpy scikit-learn matplotlib jupyter joblib
jupyter notebook notebooks/training.ipynb
```

Run all cells in order. Models will be saved as `.pkl` files in the working directory.

### Running the Streamlit app

```bash
pip install streamlit
cd streamlit_app
streamlit run app.py
```

### Live Demo

[View deployed app](https://network-anomaly-detection-kzrxbzpiac3wbmqmubrt3a.streamlit.app)

---

## Requirements

- Python 3.11+
- pandas, numpy, scikit-learn, matplotlib, joblib, streamlit
- ~8GB RAM recommended for full dataset training

---

## License

For academic assessment purposes only.
