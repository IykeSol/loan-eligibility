"""CrediSense — Intelligent Loan & Credit Risk Assessment.

Hybrid decision engine: machine learning + forward-chaining expert rules,
with Gemini-generated plain-language advisory notes.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import gemini_advisor
from src.data_loader import load_dataset
from src.ml_model import load_model, predict_probability, train
from src.preprocessing import applicant_to_frame, prepare
from src.rule_engine import forward_chain

st.set_page_config(
    page_title="CrediSense — Loan Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {font-size: 2.3rem; font-weight: 800; margin-bottom: 0;
                 background: linear-gradient(90deg, #1565c0, #00838f);
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .subtitle {color: #607d8b; font-size: 1.05rem; margin-top: 0.2rem;}
    .decision-card {border-radius: 14px; padding: 1.4rem 1.6rem; color: white; margin: 0.5rem 0;}
    .decision-approve {background: linear-gradient(135deg, #2e7d32, #43a047);}
    .decision-review  {background: linear-gradient(135deg, #ef6c00, #ffa000);}
    .decision-decline {background: linear-gradient(135deg, #b71c1c, #e53935);}
    .decision-card h2 {margin: 0; font-size: 1.7rem;}
    .decision-card p {margin: 0.3rem 0 0 0; opacity: 0.92;}
    .rule-chip {display: inline-block; border-radius: 8px; padding: 2px 10px;
                font-size: 0.8rem; font-weight: 600; margin-left: 8px;}
    .chip-up {background: #ffebee; color: #c62828;}
    .chip-down {background: #e8f5e9; color: #2e7d32;}
    .advice-box {background: #f3f7fb; border-left: 5px solid #1565c0;
                 border-radius: 10px; padding: 1.1rem 1.3rem; line-height: 1.6;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Fetching loan portfolio data...")
def get_data():
    df = load_dataset()
    return df, prepare(df)


@st.cache_resource(show_spinner="Preparing the decision engine...")
def get_model():
    bundle = load_model()
    if bundle is None:
        df, _ = get_data()
        train(df)
        bundle = load_model()
    return bundle


def risk_gauge(score: int) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100"},
            title={"text": "Credit Risk Score", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#263238", "thickness": 0.25},
                "steps": [
                    {"range": [0, 35], "color": "#a5d6a7"},
                    {"range": [35, 60], "color": "#ffe082"},
                    {"range": [60, 100], "color": "#ef9a9a"},
                ],
                "threshold": {"line": {"color": "#b71c1c", "width": 3}, "value": score},
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=50, b=10))
    return fig


def probability_donut(prob: float) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            values=[prob, 1 - prob],
            labels=["Approval likelihood", ""],
            hole=0.72,
            marker_colors=["#1565c0", "#eceff1"],
            textinfo="none",
            sort=False,
        )
    )
    fig.add_annotation(text=f"<b>{prob:.0%}</b>", font_size=30, showarrow=False)
    fig.update_layout(
        height=260, showlegend=False, margin=dict(l=30, r=30, t=50, b=10),
        title={"text": "ML Approval Likelihood", "font": {"size": 16}, "x": 0.5},
    )
    return fig


def page_assessment():
    st.markdown('<p class="main-title">💳 CrediSense</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Intelligent loan &amp; credit risk assessment — machine learning, '
        "expert rules and AI guidance working together.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    with st.form("applicant_form"):
        st.subheader("👤 Applicant Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3"])
        with c2:
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self-employed", ["No", "Yes"])
            property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
        with c3:
            credit_history = st.selectbox(
                "Credit history", ["Clean record", "Past issues"],
                help="Has the applicant repaid previous debts on time?",
            )

        st.subheader("💰 Financials")
        c4, c5, c6, c7 = st.columns(4)
        with c4:
            applicant_income = st.number_input("Applicant income / month", 0, 100000, 5000, step=250)
        with c5:
            coapplicant_income = st.number_input("Co-applicant income / month", 0, 100000, 0, step=250)
        with c6:
            loan_amount = st.number_input("Loan amount (thousands)", 1, 800, 120, step=5)
        with c7:
            term = st.selectbox("Term (months)", [360, 300, 240, 180, 120, 84, 60, 36, 480], index=0)

        submitted = st.form_submit_button("🔍 Assess Application", use_container_width=True, type="primary")

    if not submitted:
        st.info("Fill in the applicant's profile and press **Assess Application** to run the hybrid engine.")
        return

    applicant = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": term,
        "Credit_History": 1.0 if credit_history == "Clean record" else 0.0,
        "Property_Area": property_area,
    }

    bundle = get_model()
    frame = applicant_to_frame(applicant)
    ml_prob = predict_probability(bundle, frame)
    rule_result = forward_chain(applicant)

    # Blend: ML likelihood and rule-based safety on the same scale
    blended = 0.55 * ml_prob + 0.45 * (1 - rule_result.risk_score / 100)
    if blended >= 0.65 and rule_result.risk_band != "High":
        decision, css = "Approve", "decision-approve"
        tagline = "This application meets the criteria for approval."
        icon = "✅"
    elif blended >= 0.45:
        decision, css = "Refer for Review", "decision-review"
        tagline = "Borderline profile — a credit officer should take a closer look."
        icon = "🔎"
    else:
        decision, css = "Decline", "decision-decline"
        tagline = "The risk indicators outweigh the strengths of this application."
        icon = "⛔"

    st.markdown(
        f'<div class="decision-card {css}"><h2>{icon} {decision}</h2>'
        f"<p>{tagline} &nbsp;•&nbsp; Confidence blend: {blended:.0%} &nbsp;•&nbsp; "
        f"Risk band: {rule_result.risk_band}</p></div>",
        unsafe_allow_html=True,
    )

    g1, g2, g3 = st.columns([1.1, 1.1, 1])
    with g1:
        st.plotly_chart(probability_donut(ml_prob), use_container_width=True)
    with g2:
        st.plotly_chart(risk_gauge(rule_result.risk_score), use_container_width=True)
    with g3:
        st.markdown("#### Key numbers")
        total_income = applicant_income + coapplicant_income
        emi = loan_amount * 1000 / term
        st.metric("Household income", f"{total_income:,.0f} / mo")
        st.metric("Estimated repayment", f"{emi:,.0f} / mo")
        st.metric("Repayment burden", f"{emi / total_income:.0%}" if total_income else "—")

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("#### 🧠 Expert-System Reasoning (forward chaining)")
        st.caption(
            "The engine starts from the facts of the application and keeps firing rules "
            "until nothing new can be inferred. Every step below is one fired rule."
        )
        for i, (_, description, delta) in enumerate(rule_result.fired, 1):
            chip_class = "chip-up" if delta > 0 else "chip-down"
            arrow = f"+{delta} risk" if delta > 0 else f"{delta} risk"
            st.markdown(
                f"**{i}.** {description} "
                f'<span class="rule-chip {chip_class}">{arrow}</span>',
                unsafe_allow_html=True,
            )
        if not rule_result.fired:
            st.write("No rules fired — a fully neutral profile.")

    with right:
        st.markdown("#### 🤖 AI Credit Advisor")
        if gemini_advisor.is_configured():
            with st.spinner("The AI advisor is writing a personalised note..."):
                advice = gemini_advisor.generate_advice(applicant, ml_prob, rule_result, decision)
            if advice:
                st.markdown(f'<div class="advice-box">{advice}</div>', unsafe_allow_html=True)
            else:
                st.warning("The AI advisor is temporarily unavailable — the assessment above is complete without it.")
        else:
            st.info(
                "Add `GEMINI_API_KEY` to your environment or Streamlit secrets to unlock "
                "personalised AI advisory notes."
            )


def page_insights():
    st.markdown('<p class="main-title">📊 Portfolio Insights</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">What 614 real loan applications reveal about who gets approved and why.</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    df, data = get_data()
    data = data.copy()
    data["Outcome"] = np.where(df["Loan_Status"] == "Y", "Approved", "Rejected")

    approved = (df["Loan_Status"] == "Y").mean()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Applications", f"{len(df):,}")
    m2.metric("Approval rate", f"{approved:.0%}")
    m3.metric("Median income", f"{data['TotalIncome'].median():,.0f}")
    m4.metric("Median loan", f"{data['LoanAmount'].median() * 1000:,.0f}")

    st.markdown("### The credit history effect")
    st.caption("Nothing predicts the outcome like repayment history — the gap is dramatic.")
    ch = (
        data.assign(History=np.where(data["Credit_History"] >= 1, "Clean record", "Past issues"))
        .groupby(["History", "Outcome"]).size().reset_index(name="Count")
    )
    fig = px.bar(
        ch, x="History", y="Count", color="Outcome", barmode="group",
        color_discrete_map={"Approved": "#2e7d32", "Rejected": "#c62828"},
    )
    fig.update_layout(height=380, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Income vs loan size")
        st.caption("Each dot is one application. Hover to explore.")
        sample = data[data["TotalIncome"] < 25000]
        fig = px.scatter(
            sample, x="TotalIncome", y="LoanAmount", color="Outcome",
            color_discrete_map={"Approved": "#2e7d32", "Rejected": "#c62828"},
            opacity=0.55, labels={"TotalIncome": "Household income", "LoanAmount": "Loan (thousands)"},
        )
        fig.update_layout(height=380, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("### Approval rate by property area")
        st.caption("Semiurban applicants enjoy the highest approval rates.")
        pa = data.groupby("Property_Area")["Outcome"].apply(lambda s: (s == "Approved").mean() * 100).reset_index(name="Approval %")
        fig = px.bar(pa, x="Property_Area", y="Approval %", color="Approval %", color_continuous_scale="Teal")
        fig.update_layout(height=380, coloraxis_showscale=False, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### Repayment burden and outcomes")
        st.caption("Applications where repayment eats a big share of income fail more often.")
        tmp = data.copy()
        tmp["Burden"] = pd.cut(
            tmp["EMI"] / tmp["TotalIncome"].replace(0, np.nan),
            bins=[0, 0.05, 0.1, 0.2, 1.0],
            labels=["Light (<5%)", "Comfortable (5-10%)", "Noticeable (10-20%)", "Heavy (>20%)"],
        )
        bt = tmp.groupby("Burden", observed=True)["Outcome"].apply(lambda s: (s == "Approved").mean() * 100).reset_index(name="Approval %")
        fig = px.bar(bt, x="Burden", y="Approval %", color="Approval %", color_continuous_scale="Sunset")
        fig.update_layout(height=380, coloraxis_showscale=False, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        st.markdown("### Household income by outcome")
        st.caption("Approved and rejected applicants overlap far more than you would expect.")
        fig = px.box(
            data[data["TotalIncome"] < 20000], x="Outcome", y="TotalIncome", color="Outcome",
            color_discrete_map={"Approved": "#2e7d32", "Rejected": "#c62828"},
        )
        fig.update_layout(height=380, showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### What the numbers say")
    st.markdown(
        """
        - **Credit history dominates.** Applicants with a clean record are approved roughly four times
          as often as those with past issues — no other single factor comes close.
        - **Income alone doesn't decide.** Plenty of high earners are rejected and modest earners approved;
          it is the *balance* between income and repayment that matters.
        - **Semiurban properties lead approvals**, suggesting healthy demand and collateral values outside
          the big-city premium.
        - **Co-applicants help.** A second income stream visibly shifts borderline applications toward approval.
        """
    )


def page_how_it_works():
    st.markdown('<p class="main-title">⚙️ How CrediSense Decides</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Three independent minds, one transparent decision.</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🌲 Machine Learning")
        st.markdown(
            """
            A **Random Forest** trained on hundreds of real historical applications estimates the
            likelihood that a profile like this one gets approved. It sees subtle statistical
            patterns — interactions between income, loan size, term and history — that no
            hand-written rule could capture.
            """
        )
    with c2:
        st.markdown("### 🧠 Expert Rules")
        st.markdown(
            """
            A **forward-chaining inference engine** encodes how a seasoned credit officer thinks.
            Starting from the facts of the application, it fires rules, derives new facts
            (e.g. *stretched budget* + *fragile income* → *compound risk*) and keeps chaining until
            nothing new follows. Every conclusion is fully traceable.
            """
        )
    with c3:
        st.markdown("### 🤖 AI Advisor")
        st.markdown(
            """
            **Gemini** turns the technical outcome into a warm, plain-language note for the
            applicant — what helped, what hurt and what to do next. It explains; it never
            overrides the decision.
            """
        )

    st.divider()
    st.markdown("### The decision blend")
    st.markdown(
        """
        The two analytical engines are deliberately different: the model is *statistical*, the rules are
        *causal and explainable*. CrediSense blends them — 55% model likelihood, 45% rule-based safety —
        and applies a guardrail: no application in the **High** risk band is auto-approved, no matter what
        the statistics say.

        | Blended confidence | Outcome |
        |---|---|
        | ≥ 65% (and not high-risk) | ✅ Approve |
        | 45% – 65% | 🔎 Refer for human review |
        | < 45% | ⛔ Decline |

        This mirrors how modern lenders actually operate: automation for the clear cases,
        human judgement for the grey zone, and a full audit trail for every decision.
        """
    )


PAGES = {
    "💳 Risk Assessment": page_assessment,
    "📊 Portfolio Insights": page_insights,
    "⚙️ How It Works": page_how_it_works,
}

with st.sidebar:
    st.markdown("## CrediSense")
    st.caption("Intelligent Loan & Credit Risk Assessment")
    choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    bundle = load_model()
    if bundle:
        st.caption(f"Decision engine: **{bundle['name']}** + 16 expert rules")
    st.caption("AI advisor: " + ("🟢 connected" if gemini_advisor.is_configured() else "⚪ not configured"))

PAGES[choice]()
