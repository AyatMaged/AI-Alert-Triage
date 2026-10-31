# AI-Powered Cybersecurity Alert Triage System — Documentation

## 1. Project Overview
Security teams receive thousands of alerts every day. Analysts read each one manually to decide whether it is a real threat, which is slow and causes alert fatigue, so urgent cases can be missed.
This system uses **NLP and Machine Learning** to read the textual description of a security event and its evidence, classify it automatically, assign a priority, and show the results in an **interactive dashboard**.

**Dataset:** `tegridydev/opensec-triage` (Hugging Face, CC BY 4.0). Synthetic English alerts: 50,000 rows (40k train / 5k validation / 5k test), 7 classes.
**Classes:** malicious, suspicious, benign, expected_admin, authorised_testing, misconfiguration, unknown.

## 2. Project Planning
**Goal:** reduce manual triage time by auto-classifying alerts and surfacing the urgent ones first.

| Phase | Tasks | Output |
|---|---|---|
| 1. Requirements | Define problem, users, classes | Scope document |
| 2. Data | Load and explore the dataset, check class balance and text length | Clean data, EDA |
| 3. Modeling | TF-IDF (1–3 grams) + Logistic Regression, evaluate on test set | Trained model, metrics |
| 4. Dashboard | Streamlit app: single alert, batch upload, metrics | Working UI |
| 5. Testing | Test predictions, batch files, edge cases | Test results |
| 6. Delivery | GitHub repo, documentation, presentation, demo video | Final submission |

**Tools:** Python, pandas, scikit-learn, Streamlit, Plotly, Hugging Face datasets, GitHub.
**Risks:** (1) the dataset is synthetic, so real alerts may look different. Mitigation: state this limitation clearly and keep the analyst in the loop. (2) Overfitting. Mitigation: separate test split, report macro F1. (3) Class confusion between suspicious and unknown. Mitigation: show confidence and send low-confidence cases to review.

## 3. Stakeholder Analysis
| Stakeholder | Role | Needs | Influence |
|---|---|---|---|
| SOC Analyst (primary user) | Reviews alerts daily | Fast sorting, clear label, confidence, recommended action | High |
| SOC Manager | Runs the team | Metrics, reduced workload, fewer missed incidents | High |
| Incident Response team | Handles real incidents | Only high-quality escalations, no noise | Medium |
| IT / System owners | Own the assets | Notification of misconfigurations | Medium |
| CISO / Management | Decision makers | Risk reduction and cost saving | High |
| Compliance / Audit | Review process | Traceable decisions and logs | Low–Medium |
| Project team / Supervisors | Build and evaluate | Working, documented system | High |

## 4. System Design
```
Alert text (+ evidence)
        |
  TF-IDF vectorizer (1-3 grams)
        |
  Logistic Regression  ->  class + probability (confidence)
        |
  Priority mapping rule  ->  Critical / High / Medium / Low + recommended action
        |
  Streamlit dashboard (single alert, batch, model performance)
```

**Priority mapping**
| Predicted class | Priority | Recommended action |
|---|---|---|
| malicious | Critical | Escalate to incident response |
| suspicious | High | Investigate, collect evidence |
| unknown | Medium | Analyst review (evidence is insufficient) |
| misconfiguration | Medium | Notify owner to fix |
| expected_admin | Low | Close with note |
| authorised_testing | Low | Close with note |
| benign | Low | Close |

## 5. Database Design
The prototype reads JSONL/CSV files. For a production deployment the following relational schema (PostgreSQL/SQLite) is proposed.

**alerts**
| Column | Type | Notes |
|---|---|---|
| alert_id | INTEGER PK | |
| received_at | TIMESTAMP | |
| source | VARCHAR | SIEM, EDR, firewall... |
| text | TEXT | description + evidence |

**predictions**
| Column | Type | Notes |
|---|---|---|
| prediction_id | INTEGER PK | |
| alert_id | INTEGER FK -> alerts | |
| model_version | VARCHAR | |
| predicted_class | VARCHAR | one of the 7 classes |
| confidence | FLOAT | 0–1 |
| priority | VARCHAR | Critical/High/Medium/Low |
| created_at | TIMESTAMP | |

**analyst_reviews**
| Column | Type | Notes |
|---|---|---|
| review_id | INTEGER PK | |
| alert_id | INTEGER FK -> alerts | |
| analyst_id | INTEGER FK -> users | |
| final_class | VARCHAR | analyst decision |
| comment | TEXT | |
| reviewed_at | TIMESTAMP | |

**users**: user_id PK, name, role (analyst / manager), email.
**model_runs**: run_id PK, model_version, trained_at, accuracy, f1_macro, data_source.

Relations: one alert has many predictions and many reviews; analyst reviews give feedback labels for retraining.

## 6. UI/UX Design
**Users:** SOC analysts under time pressure. **Principles:** most urgent first, minimal clicks, color-coded priority, always show confidence.

**Screens (3 tabs):**
1. **Single alert** — text box, "Classify" button, result cards (class, priority, confidence), recommended action, probability bar chart.
2. **Batch triage** — file upload, KPI cards (total, critical, high, low), bar chart by priority, pie chart by class, filterable table sorted by priority, CSV download.
3. **Model performance** — accuracy, macro F1, train/test sizes, confusion matrix.

**Color code:** Critical red, High orange, Medium yellow, Low green.
**Wireframe (batch tab):**
```
+--------------------------------------------------+
| AI-Powered Cybersecurity Alert Triage            |
| [Single] [Batch] [Performance]                   |
| [ Upload file ]                                  |
| Total | Critical | High | Low                    |
| [Bar: priority]        [Pie: class]              |
| Filter: [Critical][High]                         |
| Table: text | class | confidence | priority |... |
| [Download CSV]                                   |
+--------------------------------------------------+
```

## 7. Results and Limitations
Metrics are saved to `models/metrics.json` and shown in the dashboard. The dataset is synthetic and its evidence sentences state the outcome explicitly, so very high scores are expected. They do **not** prove the same accuracy on real SOC alerts. Future work: real logs, transformer models (e.g. DistilBERT), analyst feedback loop, explainability of the decision.
