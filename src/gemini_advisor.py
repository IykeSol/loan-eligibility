"""Gemini-powered natural-language credit advisory.

Reads GEMINI_API_KEY / GEMINI_MODEL from Streamlit secrets or environment
variables (a local .env is also supported). Fails soft: if the API is
unavailable the app still works without the AI narrative.
"""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional — secrets/env vars still work
    pass


def _get_config():
    api_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
    try:
        import streamlit as st

        api_key = st.secrets.get("GEMINI_API_KEY", api_key)
        model = st.secrets.get("GEMINI_MODEL", model)
    except Exception:  # noqa: BLE001 - secrets file may not exist
        pass
    return api_key, model


def is_configured() -> bool:
    key, _ = _get_config()
    return bool(key)


def generate_advice(applicant: dict, ml_probability: float, rule_result, final_decision: str) -> str:
    """Ask Gemini to explain the assessment in plain language with actionable advice."""
    api_key, model_name = _get_config()
    if not api_key:
        return ""

    fired_rules = "\n".join(f"- {desc} (impact {delta:+d})" for _, desc, delta in rule_result.fired)

    prompt = f"""You are a friendly senior credit analyst at a digital lender.
An applicant was just assessed by a hybrid system (machine learning + expert rules).

Applicant profile:
- Gender: {applicant.get('Gender')}, Married: {applicant.get('Married')}, Dependents: {applicant.get('Dependents')}
- Education: {applicant.get('Education')}, Self-employed: {applicant.get('Self_Employed')}
- Applicant income: {applicant.get('ApplicantIncome')} /month, Co-applicant income: {applicant.get('CoapplicantIncome')} /month
- Loan requested: {float(applicant.get('LoanAmount', 0)) * 1000:,.0f} over {applicant.get('Loan_Amount_Term')} months
- Credit history clean: {'Yes' if float(applicant.get('Credit_History', 0)) >= 1 else 'No'}
- Property area: {applicant.get('Property_Area')}

Assessment outcome:
- ML approval probability: {ml_probability:.0%}
- Rule-based risk score: {rule_result.risk_score}/100 ({rule_result.risk_band} risk)
- Final decision: {final_decision}

Reasoning trace from the expert system:
{fired_rules}

Write a short, warm, professional note (max 180 words) addressed to the applicant:
1. One-sentence summary of the outcome.
2. The 2-3 factors that mattered most, in plain English.
3. Concrete, encouraging next steps to improve or maintain their standing.
Do not mention internal scores, probabilities, rule names, or system details. No headings, just 2-3 short paragraphs."""

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model_name, contents=prompt)
        return (response.text or "").strip()
    except Exception:  # noqa: BLE001 - fall through to legacy SDK
        pass

    try:
        import google.generativeai as legacy_genai

        legacy_genai.configure(api_key=api_key)
        model = legacy_genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return (response.text or "").strip()
    except Exception:  # noqa: BLE001 - AI narrative is optional
        return ""
