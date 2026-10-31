"""Interactive analyst dashboard for the AI-Powered Cybersecurity Alert Triage System.

Run:  streamlit run app.py
"""
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AI Alert Triage", page_icon="🛡️", layout="wide")

# Disposition -> analyst action / priority
PRIORITY = {
    "malicious": ("Critical", "Escalate to incident response now"),
    "suspicious": ("High", "Investigate and gather more evidence"),
    "unknown": ("Medium", "Insufficient evidence - analyst review required"),
    "misconfiguration": ("Medium", "Notify owner to fix the configuration"),
    "expected_admin": ("Low", "Expected admin activity - close with note"),
    "authorised_testing": ("Low", "Approved security test - close with note"),
    "benign": ("Low", "No action needed - close"),
}
PRIORITY_ORDER = ["Critical", "High", "Medium", "Low"]
COLORS = {"Critical": "#d62728", "High": "#ff7f0e", "Medium": "#f2c037", "Low": "#2ca02c"}


@st.cache_resource
def load_model():
    return joblib.load("models/triage_model.joblib")


@st.cache_data
def load_metrics():
    with open("models/metrics.json") as f:
        return json.load(f)


def triage(texts):
    model = load_model()
    proba = model.predict_proba(texts)
    classes = model.classes_
    out = pd.DataFrame({"text": list(texts)})
    out["prediction"] = [classes[i] for i in proba.argmax(axis=1)]
    out["confidence"] = proba.max(axis=1).round(3)
    out["priority"] = out["prediction"].map(lambda c: PRIORITY[c][0])
    out["recommended_action"] = out["prediction"].map(lambda c: PRIORITY[c][1])
    out["_rank"] = out["priority"].map(PRIORITY_ORDER.index)
    out = out.sort_values(["_rank", "confidence"], ascending=[True, False]).drop(columns="_rank")
    return out.reset_index(drop=True), proba, classes


def read_upload(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    else:  # .jsonl / .txt with JSON lines
        df = pd.read_json(file, lines=True)
    if "text" not in df.columns:
        st.error("The file must contain a 'text' column.")
        st.stop()
    return df


st.title("🛡️ AI-Powered Cybersecurity Alert Triage")
st.caption("NLP + Machine Learning classify each alert and rank it by priority for the analyst.")

tab1, tab2, tab3 = st.tabs(["🔍 Single alert", "📂 Batch triage", "📊 Model performance"])

# ---------------- Single alert ----------------
with tab1:
    sample = (
        "Ssh audit on LEGAL-ASSET-b15f847f44eda9b5 recorded that an SSH session opened and requested "
        "elevated command execution. The event references principal-17ee862a652063ef. "
        "A collection gap removed the surrounding events needed to determine the action's purpose."
    )
    text = st.text_area("Paste the alert description and evidence", value=sample, height=180)
    if st.button("Classify alert", type="primary"):
        res, proba, classes = triage([text])
        row = res.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Classification", row["prediction"])
        c2.metric("Priority", row["priority"])
        c3.metric("Confidence", f"{row['confidence']:.0%}")
        st.info(f"Recommended action: {row['recommended_action']}")
        probs = pd.DataFrame({"class": classes, "probability": proba[0]}).sort_values("probability")
        st.plotly_chart(
            px.bar(probs, x="probability", y="class", orientation="h", title="Class probabilities"),
            use_container_width=True,
        )

# ---------------- Batch ----------------
with tab2:
    file = st.file_uploader("Upload alerts (.csv or .jsonl with a 'text' column)", type=["csv", "jsonl", "txt"])
    if file is not None:
        df = read_upload(file)
        res, _, _ = triage(df["text"].astype(str).tolist())
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total alerts", len(res))
        k2.metric("Critical", int((res.priority == "Critical").sum()))
        k3.metric("High", int((res.priority == "High").sum()))
        k4.metric("Low / auto-closable", int((res.priority == "Low").sum()))

        left, right = st.columns(2)
        pc = res["priority"].value_counts().reindex(PRIORITY_ORDER).fillna(0).reset_index()
        pc.columns = ["priority", "count"]
        left.plotly_chart(
            px.bar(pc, x="priority", y="count", color="priority", color_discrete_map=COLORS, title="Alerts by priority"),
            use_container_width=True,
        )
        right.plotly_chart(
            px.pie(res, names="prediction", title="Alerts by classification"), use_container_width=True
        )

        flt = st.multiselect("Filter by priority", PRIORITY_ORDER, default=["Critical", "High"])
        view = res[res.priority.isin(flt)] if flt else res
        st.dataframe(view, use_container_width=True, height=400)
        st.download_button("Download results (CSV)", res.to_csv(index=False), "triage_results.csv", "text/csv")
    else:
        st.info("Upload a file to triage many alerts at once. Critical and High alerts are shown first.")

# ---------------- Metrics ----------------
with tab3:
    m = load_metrics()
    a, b, c = st.columns(3)
    a.metric("Accuracy", f"{m['accuracy']:.1%}")
    b.metric("Macro F1", f"{m['f1_macro']:.3f}")
    c.metric("Train / test size", f"{m['n_train']} / {m['n_test']}")
    st.caption(f"Data source: {m['data_source']}")
    cm = pd.DataFrame(m["confusion_matrix"], index=m["labels"], columns=m["labels"])
    st.plotly_chart(
        px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual"),
                  title="Confusion matrix"),
        use_container_width=True,
    )
