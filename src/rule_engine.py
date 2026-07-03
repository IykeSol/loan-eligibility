"""Rule-based credit risk assessment using forward chaining.

The engine starts from base facts derived from the applicant's data and
repeatedly fires rules whose conditions are satisfied, adding new facts to
working memory until no more rules can fire. Each fired rule contributes to
the risk score and leaves an auditable trace of the reasoning chain.
"""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Rule:
    name: str
    description: str
    condition: Callable[[set], bool]
    adds: set
    risk_delta: int = 0  # positive = riskier, negative = safer


@dataclass
class InferenceResult:
    facts: set
    fired: list = field(default_factory=list)  # ordered trace of (rule name, description, delta)
    risk_score: int = 50  # 0 (safest) .. 100 (riskiest)
    risk_band: str = "Medium"
    decision: str = "Review"


def derive_base_facts(applicant: dict) -> set:
    """Translate raw applicant values into symbolic facts for the engine."""
    facts = set()

    total_income = float(applicant.get("ApplicantIncome", 0)) + float(applicant.get("CoapplicantIncome", 0))
    loan_amount = float(applicant.get("LoanAmount", 0)) * 1000  # dataset stores thousands
    term = float(applicant.get("Loan_Amount_Term", 360)) or 360
    emi = loan_amount / term if term else 0
    dti = emi / total_income if total_income else 1.0

    if float(applicant.get("Credit_History", 0)) >= 1:
        facts.add("good_credit_history")
    else:
        facts.add("bad_credit_history")

    if total_income >= 10000:
        facts.add("high_income")
    elif total_income >= 4000:
        facts.add("moderate_income")
    else:
        facts.add("low_income")

    if float(applicant.get("CoapplicantIncome", 0)) > 0:
        facts.add("has_coapplicant")

    if dti <= 0.25:
        facts.add("low_dti")
    elif dti <= 0.45:
        facts.add("moderate_dti")
    else:
        facts.add("high_dti")

    if loan_amount > 0 and total_income > 0 and loan_amount / (total_income * 12) > 5:
        facts.add("oversized_loan")

    if str(applicant.get("Education", "")) == "Graduate":
        facts.add("graduate")
    if str(applicant.get("Self_Employed", "No")) == "Yes":
        facts.add("self_employed")
    if str(applicant.get("Married", "No")) == "Yes":
        facts.add("married")

    area = str(applicant.get("Property_Area", "Urban")).lower()
    facts.add(f"property_{area}")

    deps = str(applicant.get("Dependents", "0")).replace("+", "")
    try:
        if int(float(deps)) >= 3:
            facts.add("many_dependents")
    except ValueError:
        pass

    if term <= 180:
        facts.add("short_term_loan")
    elif term >= 480:
        facts.add("very_long_term_loan")

    return facts


RULES = [
    Rule(
        "R1_credit_anchor",
        "A clean repayment history is the strongest single indicator of reliability.",
        lambda f: "good_credit_history" in f,
        {"reliable_payer"},
        -18,
    ),
    Rule(
        "R2_credit_red_flag",
        "Past defaults or missed payments raise the chance of future default sharply.",
        lambda f: "bad_credit_history" in f,
        {"default_risk"},
        +28,
    ),
    Rule(
        "R3_comfortable_repayment",
        "Monthly repayment takes only a small share of income, leaving a wide safety margin.",
        lambda f: "low_dti" in f,
        {"strong_affordability"},
        -14,
    ),
    Rule(
        "R4_stretched_budget",
        "Repayment consumes a large share of monthly income, leaving little buffer for shocks.",
        lambda f: "high_dti" in f,
        {"weak_affordability"},
        +18,
    ),
    Rule(
        "R5_income_strength",
        "A high household income cushions the loan against unexpected expenses.",
        lambda f: "high_income" in f,
        {"income_buffer"},
        -10,
    ),
    Rule(
        "R6_thin_income",
        "Low household income offers little room to absorb financial shocks.",
        lambda f: "low_income" in f,
        {"income_fragility"},
        +12,
    ),
    Rule(
        "R7_shared_responsibility",
        "A co-applicant adds a second income stream and shared accountability.",
        lambda f: "has_coapplicant" in f and "income_fragility" not in f,
        {"joint_backing"},
        -6,
    ),
    Rule(
        "R8_oversized_request",
        "The requested amount is very large relative to yearly income.",
        lambda f: "oversized_loan" in f,
        {"exposure_concern"},
        +14,
    ),
    Rule(
        "R9_compound_strain",
        "Weak affordability combined with fragile income compounds the strain.",
        lambda f: "weak_affordability" in f and "income_fragility" in f,
        {"compound_risk"},
        +10,
    ),
    Rule(
        "R10_dependents_load",
        "Several dependents on a stretched budget reduce disposable income further.",
        lambda f: "many_dependents" in f and ("weak_affordability" in f or "income_fragility" in f),
        {"household_pressure"},
        +6,
    ),
    Rule(
        "R11_stable_profile",
        "A reliable payer with strong affordability is a textbook low-risk borrower.",
        lambda f: "reliable_payer" in f and "strong_affordability" in f,
        {"prime_borrower"},
        -8,
    ),
    Rule(
        "R12_self_employment_variance",
        "Self-employed income tends to fluctuate, adding mild uncertainty.",
        lambda f: "self_employed" in f and "income_buffer" not in f,
        {"income_variance"},
        +5,
    ),
    Rule(
        "R13_education_signal",
        "Graduate education correlates with steadier long-term earning power.",
        lambda f: "graduate" in f,
        {"earning_potential"},
        -3,
    ),
    Rule(
        "R14_long_horizon",
        "A very long repayment horizon extends the window in which things can go wrong.",
        lambda f: "very_long_term_loan" in f,
        {"extended_exposure"},
        +4,
    ),
    Rule(
        "R15_semiurban_edge",
        "Semiurban properties historically show the healthiest repayment rates in this portfolio.",
        lambda f: "property_semiurban" in f,
        {"favourable_market"},
        -4,
    ),
    Rule(
        "R16_redemption_path",
        "Even with a weak history, strong affordability and high income can carry the loan.",
        lambda f: "default_risk" in f and "strong_affordability" in f and "income_buffer" in f,
        {"mitigated_risk"},
        -8,
    ),
]


def forward_chain(applicant: dict) -> InferenceResult:
    """Run forward chaining until no new facts can be inferred."""
    facts = derive_base_facts(applicant)
    result = InferenceResult(facts=facts)
    fired_names = set()

    changed = True
    while changed:
        changed = False
        for rule in RULES:
            if rule.name in fired_names:
                continue
            if rule.condition(facts):
                facts |= rule.adds
                fired_names.add(rule.name)
                result.fired.append((rule.name, rule.description, rule.risk_delta))
                result.risk_score += rule.risk_delta
                changed = True

    result.risk_score = max(0, min(100, result.risk_score))

    if result.risk_score < 35:
        result.risk_band, result.decision = "Low", "Approve"
    elif result.risk_score < 60:
        result.risk_band, result.decision = "Medium", "Review"
    else:
        result.risk_band, result.decision = "High", "Decline"

    return result
