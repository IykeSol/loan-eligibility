# 💳 CrediSense — Intelligent Loan & Credit Risk Assessment

A hybrid credit-decision system that combines **machine learning**, a **forward-chaining
rule-based expert system**, and **Gemini AI** advisory notes — wrapped in a polished
Streamlit experience.

## ✨ What it does

| Engine | Role |
|---|---|
| 🌲 **Random Forest** (scikit-learn) | Learns approval patterns from 614 real historical loan applications |
| 🧠 **Forward-chaining expert system** | 16 hand-crafted credit rules fire iteratively, deriving new facts until a fully traceable risk score emerges |
| 🤖 **Gemini AI advisor** | Turns the technical outcome into a warm, plain-language note with actionable next steps |

The two analytical engines are blended (55% ML likelihood, 45% rule-based safety) with a
guardrail: high-risk profiles are never auto-approved.

## 🚀 Run it locally

```bash
git clone https://github.com/IykeSol/loan-eligibility.git
cd loan-eligibility
pip install -r requirements.txt

# optional — unlocks the AI advisor
echo GEMINI_API_KEY=your_key_here > .env
echo GEMINI_MODEL=gemini-flash-lite-latest >> .env

streamlit run app.py
```

The dataset is fetched online automatically on first run and cached in `data/`.
A pre-trained model ships in `models/`; retrain any time with:

```bash
python train.py
```

`train.py` also regenerates every chart in `reports/figures/`.

## ☁️ Deploy on Streamlit Community Cloud

1. Fork / push this repo to GitHub.
2. Create a new app at [share.streamlit.io](https://share.streamlit.io) pointing at `app.py`.
3. In **App settings → Secrets**, add:

```toml
GEMINI_API_KEY = "your_key_here"
GEMINI_MODEL = "gemini-flash-lite-latest"
```

That's it — the app trains itself on first boot if no model artifact is found.

## 🗂️ Project layout

```
app.py                  Streamlit app (assessment, insights, how-it-works)
train.py                Trains models + generates all figures
src/
  data_loader.py        Online dataset fetch with local cache
  preprocessing.py      Cleaning + affordability feature engineering
  ml_model.py           Candidate models, cross-validation, persistence
  rule_engine.py        Forward-chaining inference engine (16 rules)
  gemini_advisor.py     Gemini plain-language advisory notes
models/                 Trained model artifact
reports/figures/        Generated charts
data/                   Cached dataset
```

## 🔍 The forward-chaining engine in 20 seconds

Applicant data is converted into symbolic **facts** (`good_credit_history`,
`high_dti`, `low_income`, …). Rules whose conditions match the working memory
fire, **add new facts** (`weak_affordability`, `compound_risk`, …) and adjust the
risk score. The cycle repeats until no rule can fire — classic forward chaining —
and the full trace is displayed with every assessment, so every decision is
explainable end-to-end.
