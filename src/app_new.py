"""
CreditGuard — interactive version.

Run with: streamlit run src/app.py

Adds: a Plotly risk gauge, sliders instead of plain number boxes,
colored metric cards, a sidebar for inputs (more app-like layout),
and a cleaner SHAP explanation section with a plain-language summary.

Expects a trained model saved via joblib as ../model.pkl.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go

st.set_page_config(
    page_title="CreditGuard — Loan Risk Score",
    page_icon="💳",
    layout="wide",
)

MODEL_PATH = "../model.pkl"
BACKGROUND_PATH = "../background.pkl"

FEATURE_COLUMNS = [
    'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
    'revol_bal', 'revol_util', 'term_ 60 months',
    'grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G',
    'sub_grade_A2', 'sub_grade_A3', 'sub_grade_A4', 'sub_grade_A5',
    'sub_grade_B1', 'sub_grade_B2', 'sub_grade_B3', 'sub_grade_B4', 'sub_grade_B5',
    'sub_grade_C1', 'sub_grade_C2', 'sub_grade_C3', 'sub_grade_C4', 'sub_grade_C5',
    'sub_grade_D1', 'sub_grade_D2', 'sub_grade_D3', 'sub_grade_D4', 'sub_grade_D5',
    'sub_grade_E1', 'sub_grade_E2', 'sub_grade_E3', 'sub_grade_E4', 'sub_grade_E5',
    'sub_grade_F1', 'sub_grade_F2', 'sub_grade_F3', 'sub_grade_F4', 'sub_grade_F5',
    'sub_grade_G1', 'sub_grade_G2', 'sub_grade_G3', 'sub_grade_G4', 'sub_grade_G5',
    'emp_length_10+ years', 'emp_length_2 years', 'emp_length_3 years',
    'emp_length_4 years', 'emp_length_5 years', 'emp_length_6 years',
    'emp_length_7 years', 'emp_length_8 years', 'emp_length_9 years',
    'emp_length_< 1 year', 'emp_length_missing',
    'home_ownership_MORTGAGE', 'home_ownership_NONE', 'home_ownership_OTHER',
    'home_ownership_OWN', 'home_ownership_RENT',
    'verification_status_Source Verified', 'verification_status_Verified',
    'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational',
    'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase',
    'purpose_medical', 'purpose_moving', 'purpose_other',
    'purpose_renewable_energy', 'purpose_small_business', 'purpose_vacation',
    'purpose_wedding',
]

GRADES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
SUB_GRADE_NUMS = ['1', '2', '3', '4', '5']
EMP_LENGTHS = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years',
               '6 years', '7 years', '8 years', '9 years', '10+ years', 'missing']
HOME_OWNERSHIP = ['RENT', 'MORTGAGE', 'OWN', 'OTHER', 'NONE']
VERIFICATION = ['Not Verified', 'Source Verified', 'Verified']
PURPOSES = [
    'debt_consolidation', 'credit_card', 'home_improvement', 'other',
    'major_purchase', 'medical', 'small_business', 'moving', 'vacation',
    'house', 'wedding', 'renewable_energy', 'educational',
]
TERMS = [' 36 months', ' 60 months']

# ---------- styling ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #22c55e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(148, 163, 184, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None


@st.cache_resource
def load_background():
    try:
        return joblib.load(BACKGROUND_PATH)
    except FileNotFoundError:
        return None


def build_input_row(inputs: dict) -> pd.DataFrame:
    row = {col: 0 for col in FEATURE_COLUMNS}
    row['loan_amnt'] = inputs['loan_amnt']
    row['int_rate'] = inputs['int_rate']
    row['installment'] = inputs['installment']
    row['annual_inc'] = inputs['annual_inc']
    row['dti'] = inputs['dti']
    row['revol_bal'] = inputs['revol_bal']
    row['revol_util'] = inputs['revol_util']

    if inputs['term'] == ' 60 months':
        row['term_ 60 months'] = 1

    for key, prefix in [
        ('grade', 'grade_'), ('sub_grade', 'sub_grade_'),
        ('emp_length', 'emp_length_'), ('home_ownership', 'home_ownership_'),
        ('verification_status', 'verification_status_'), ('purpose', 'purpose_'),
    ]:
        col = f"{prefix}{inputs[key]}"
        if col in row:
            row[col] = 1

    return pd.DataFrame([row])[FEATURE_COLUMNS]


def risk_gauge(proba: float) -> go.Figure:
    pct = proba * 100
    if pct < 25:
        bar_color, label = "#22c55e", "Low risk"
    elif pct < 50:
        bar_color, label = "#eab308", "Moderate risk"
    else:
        bar_color, label = "#ef4444", "High risk"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': "%", 'font': {'size': 44}},
        title={'text': label, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 25], 'color': 'rgba(34,197,94,0.15)'},
                {'range': [25, 50], 'color': 'rgba(234,179,8,0.15)'},
                {'range': [50, 100], 'color': 'rgba(239,68,68,0.15)'},
            ],
            'threshold': {
                'line': {'color': bar_color, 'width': 4},
                'thickness': 0.8,
                'value': pct,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10))
    return fig


st.markdown('<p class="main-header">💳 CreditGuard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Instant, explainable loan default risk scoring</p>', unsafe_allow_html=True)
st.write("")

model = load_model()
background = load_background()

if model is None:
    st.warning(
        f"No trained model found at `{MODEL_PATH}`. Train and save a model "
        "from `notebooks/01_eda_and_modeling.ipynb` first "
        "(`joblib.dump(best_model, '../model.pkl')`)."
    )
    st.stop()

if background is None:
    st.warning(
        f"No background sample found at `{BACKGROUND_PATH}`. In the notebook, run:\n\n"
        "```python\n"
        "background_sample = X_train_sm.sample(100, random_state=42).astype(float)\n"
        "joblib.dump(background_sample, '../background.pkl')\n"
        "```\n\nThen restart this app. SHAP needs a real background sample to compute "
        "meaningful explanations — using the single input row as its own background "
        "produces zero-impact (blank) results."
    )
    st.stop()

# ---------- sidebar inputs ----------
with st.sidebar:
    st.header("📋 Applicant Details")

    st.subheader("Loan")
    loan_amnt = st.slider("Loan amount ($)", 500, 40000, 15000, step=500)
    term = st.radio("Term", TERMS, horizontal=True)
    int_rate = st.slider("Interest rate (%)", 5.0, 30.0, 12.5, step=0.1)
    installment = st.number_input("Monthly installment ($)", min_value=0.0, value=450.0, step=10.0)
    purpose = st.selectbox("Purpose", PURPOSES)

    st.subheader("Credit profile")
    grade = st.select_slider("Grade", GRADES, value="C")
    sub_grade = st.selectbox("Sub-grade", [f"{grade}{n}" for n in SUB_GRADE_NUMS])
    revol_bal = st.slider("Revolving balance ($)", 0, 50000, 5000, step=500)
    revol_util = st.slider("Revolving utilization (%)", 0.0, 100.0, 40.0, step=1.0)

    st.subheader("Financial profile")
    annual_inc = st.slider("Annual income ($)", 0, 250000, 60000, step=1000)
    dti = st.slider("Debt-to-income (%)", 0.0, 50.0, 18.0, step=0.5)
    emp_length = st.selectbox("Employment length", EMP_LENGTHS)
    home_ownership = st.selectbox("Home ownership", HOME_OWNERSHIP)
    verification_status = st.selectbox("Income verification", VERIFICATION)

    submitted = st.button("🔍 Estimate Risk", use_container_width=True, type="primary")

# ---------- main panel ----------
if not submitted:
    st.info("👈 Fill in applicant details in the sidebar and click **Estimate Risk** to get a score.")
else:
    inputs = {
        'loan_amnt': loan_amnt, 'int_rate': int_rate, 'installment': installment,
        'annual_inc': annual_inc, 'dti': dti, 'revol_bal': revol_bal,
        'revol_util': revol_util, 'term': term, 'grade': grade,
        'sub_grade': sub_grade, 'emp_length': emp_length,
        'home_ownership': home_ownership, 'verification_status': verification_status,
        'purpose': purpose,
    }
    row = build_input_row(inputs).astype(float)
    proba = model.predict_proba(row)[0, 1]

    col_gauge, col_metrics = st.columns([1.2, 1])

    with col_gauge:
        st.plotly_chart(risk_gauge(proba), use_container_width=True)

    with col_metrics:
        st.write("")
        st.metric("Default probability", f"{proba:.1%}")
        st.metric("Loan amount", f"${loan_amnt:,.0f}")
        st.metric("Requested grade", f"{grade} ({sub_grade})")
        monthly_income = annual_inc / 12
        affordability = installment / monthly_income if monthly_income else 0
        st.metric("Installment / income", f"{affordability:.1%}")

    st.divider()

    st.subheader("🔬 Why this score?")
    explainer = shap.Explainer(model, background)
    shap_values = explainer(row)

    contributions = pd.DataFrame({
    'feature': FEATURE_COLUMNS,
    'impact': shap_values.values[0, :, 1],
})
    contributions['direction'] = np.where(contributions['impact'] > 0, 'Increases risk', 'Decreases risk')
    top = contributions.reindex(contributions['impact'].abs().sort_values(ascending=False).index).head(8)

    fig = go.Figure(go.Bar(
        x=top['impact'],
        y=top['feature'],
        orientation='h',
        marker_color=np.where(top['impact'] > 0, '#ef4444', '#22c55e'),
    ))
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Impact on risk score",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    top_driver = top.iloc[0]
    direction_word = "increased" if top_driver['impact'] > 0 else "decreased"
    st.caption(
        f"The single biggest factor was **{top_driver['feature']}**, which {direction_word} "
        f"this applicant's risk score the most. Red bars push risk up, green bars pull it down."
    )
