import streamlit as st
import pandas as pd
import joblib
import os

from dotenv import load_dotenv
from google import genai

# ======================================================
# LOAD ENV VARIABLES (.env)
# ======================================================
load_dotenv(override=True)

# ======================================================
# STREAMLIT PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Smart Water Quality Analyzer",
    layout="centered"
)

st.title("💧 Smart Water Quality Analyzer")
st.write("WHO Rule-Based Safety + ML Prediction + AI Usage Insights")

# ======================================================
# LOAD ML MODEL & SCALER
# ======================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("gb_water_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error("❌ Failed to load ML model or scaler.")
    st.exception(e)
    st.stop()

# ======================================================
# CONFIGURE GEMINI CLIENT
# ======================================================
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ GEMINI_API_KEY not found.")
    st.info("Create a .env file with: GEMINI_API_KEY=your_key_here")
    st.stop()

client = genai.Client(api_key=api_key)

# ======================================================
# FEATURE LIST (MUST MATCH TRAINING ORDER)
# ======================================================
important_features = [
    "ph",
    "turbidity",
    "conductivity",
    "dissolved_oxygen",
    "tds"
]

# ======================================================
# WHO RULE CHECK (HARD FAIL)
# ======================================================
def who_rule_check(sample):
    if sample["ph"] < 6.5 or sample["ph"] > 8.5:
        return 0
    if sample["turbidity"] >= 5:
        return 0
    if sample["conductivity"] >= 400:
        return 0
    if sample["dissolved_oxygen"] < 6.5 or sample["dissolved_oxygen"] > 8:
        return 0
    if sample["tds"] >= 400:
        return 0
    return None

# ======================================================
# HYBRID PREDICTION (WHO + ML)
# ======================================================
def hybrid_predict(sample_df, threshold=0.5):
    sample_dict = sample_df.iloc[0].to_dict()

    # WHO hard rule
    if who_rule_check(sample_dict) == 0:
        return "❌ Unsafe (WHO Rule Violation)", 0.0

    try:
        scaled_data = scaler.transform(sample_df)
        prob = model.predict_proba(scaled_data)[0][1]
    except Exception as e:
        return f"⚠️ Prediction Error: {e}", 0.0

    if prob >= threshold:
        return "✅ Safe Water", prob
    else:
        return "⚠️ Unsafe / Needs Treatment", prob

# ======================================================
# GEMINI INSIGHT GENERATOR (UNCHANGED STYLE)
# ======================================================
def generate_water_insights(sample, decision, probability):
    try:
        prompt = f"""
You are an environmental and water-treatment engineering expert.

Measured water parameters:
- pH: {sample['ph']}
- Turbidity: {sample['turbidity']} NTU
- Electrical Conductivity: {sample['conductivity']} µS/cm
- Dissolved Oxygen: {sample['dissolved_oxygen']} mg/L
- Total Dissolved Solids (TDS): {sample['tds']} ppm

System decision: {decision}
ML confidence: {probability}

Your task is to generate a structured expert report with the following sections:

### 1. Water Quality Classification
- Describe the overall quality grade of the water (e.g., potable, marginal, industrial-grade, contaminated).

### 2. Key Issues Identified
- Briefly list which parameters are out of range and why they matter.

### 3. Recommended Treatment & Filtration Methods
Suggest suitable treatment methods based on the measured parameters.
Examples (do NOT limit yourself to these):
- pH correction using lime, caustic soda, soda ash, or CO₂ dosing
- Turbidity removal using coagulation–flocculation (alum, ferric salts)
- Membrane separation (RO / UF / NF) for high EC or TDS
- Activated carbon filtration
- Aeration or oxygenation for low dissolved oxygen
- Ion exchange where applicable

Explain **why** each method is recommended.

⚠️ These are advisory engineering suggestions, not operational instructions.

### 4. Post-Treatment Usage Possibilities
After appropriate treatment, suggest suitable uses such as:
- Agricultural irrigation (mention specific crops by name)
- Horticulture or home gardening (mention plant types)
- Pisciculture or aquaculture (mention fish types if suitable)
- Industrial or non-potable reuse if applicable

### 5. Health & Environmental Considerations
- Summarize any remaining risks or precautions.

### 6. Short Conclusion
- 2 concise lines summarizing treatment feasibility and reuse potential.

Formatting rules:
- Use clear headings
- Use bullet points
- Keep language simple and practical
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"❌ AI insight generation failed.\n\n{str(e)}"


# ======================================================
# USER INPUT UI
# ======================================================
st.markdown("### 🔢 Enter Water Quality Parameters")

# Conversion factor
TDS_FACTOR = 0.64

# Initialize session state for synced inputs
if "conductivity" not in st.session_state:
    st.session_state.conductivity = 300.0
    st.session_state.tds = st.session_state.conductivity * TDS_FACTOR

# Callbacks to sync the two values
def _update_tds():
    st.session_state.tds = st.session_state.conductivity * TDS_FACTOR

def _update_conductivity():
    st.session_state.conductivity = st.session_state.tds / TDS_FACTOR

col1, col2 = st.columns(2)

with col1:
    ph = st.number_input("pH", 0.0, 14.0, 7.0)
    conductivity = st.number_input(
        "Conductivity (µS/cm)", 
        0.0, 
        5000.0,
        key="conductivity",
        on_change=_update_tds
    )
    dissolved_oxygen = st.number_input("Dissolved Oxygen (mg/L)", 0.0, 15.0, 7.0)

with col2:
    turbidity = st.number_input("Turbidity (NTU)", 0.0, 20.0, 3.0)
    tds = st.number_input(
        "TDS (ppm)",
        0.0,
        3200.0, # 5000 * 0.64
        key="tds",
        on_change=_update_conductivity
    )

st.caption(f"📌 TDS and Conductivity are linked (TDS = {TDS_FACTOR} × EC)")


# ======================================================
# MAIN ACTION
# ======================================================
if st.button("🔍 Analyze Water Quality"):

    sample_df = pd.DataFrame([[
        ph,
        turbidity,
        conductivity,
        dissolved_oxygen,
        tds
    ]], columns=important_features)

    decision, prob = hybrid_predict(sample_df)

    st.markdown("---")
    st.subheader("📊 Water Safety Assessment")

    if "✅" in decision:
        st.success(decision)
    else:
        st.error(decision)

    st.info(f"**ML Confidence:** {prob:.2%}")

    st.markdown("---")
    st.subheader("🧠 Intelligent Usage Insights")

    with st.spinner("Generating expert analysis using AI..."):
        insights = generate_water_insights(
            sample_df.iloc[0],
            decision,
            round(prob, 3)
        )

    st.markdown(insights)

    st.markdown("---")
    st.caption("⚠️ AI insights are advisory. WHO rules have priority.")
