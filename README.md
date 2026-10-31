# AI-Powered Cybersecurity Alert Triage System

NLP + Machine Learning system that reads security alert text, classifies it into 7 dispositions, assigns a priority, and shows results in a Streamlit dashboard.

Full documentation (planning, stakeholders, database design, UI/UX): see [DOCUMENTATION.md](DOCUMENTATION.md).

## Setup
```bash
pip install -r requirements.txt
```

## Train
Uses the dataset [tegridydev/opensec-triage](https://huggingface.co/datasets/tegridydev/opensec-triage) (downloaded automatically with the `datasets` library, or place `train.jsonl` / `test.jsonl` in `data/`).
```bash
python train.py
```

## Run the dashboard
```bash
streamlit run app.py
```
Tabs: single alert, batch triage (upload CSV/JSONL with a `text` column), model performance.

## Structure
```
app.py              Streamlit dashboard
train.py            Training and evaluation
data/               Dataset files (optional)
models/             Saved model and metrics
DOCUMENTATION.md    Project documentation
```
