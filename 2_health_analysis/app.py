import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

APP_DIR = Path(__file__).parent
DEFAULT_REPORT_PATH = APP_DIR / "blood_work.txt"

st.set_page_config(page_title="Blood Report Analysis", layout="wide")


@st.cache_resource
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model="gemini-3.6-flash")


def extract_values(llm: ChatGoogleGenerativeAI, blood_report: str) -> str:
    prompt = f"""
You are a medical data extraction assistant.

From the blood report below, extract ALL test values and classify each one as HIGH, LOW, or NORMAL based on the reference ranges provided in the report.

Format your response as:
- Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range

Blood Report:
{blood_report}
"""
    return llm.invoke(prompt).text


def generate_summary(llm: ChatGoogleGenerativeAI, extracted_values: str) -> str:
    prompt = f"""
You are a clinical physician explaining lab results to a patient.

Based on the blood work analysis below, write a short health summary in 3-5 lines explaining the patient's condition in simple, non-technical language.

Blood Work Analysis:
{extracted_values}
"""
    return llm.invoke(prompt).text


def generate_diet_plan(llm: ChatGoogleGenerativeAI, extracted_values: str) -> str:
    prompt = f"""
You are a clinical nutritionist specializing in Nigerian dietary habits.

Based on the blood work analysis below, write a short, practical Nigerian diet plan having only two sections:
(1) Foods to avoid
(2) Foods to eat more of
Do not include any other sections in the diet plan.

Blood Work Analysis:
{extracted_values}
"""
    return llm.invoke(prompt).text


def run_analysis(blood_report: str) -> None:
    llm = get_llm()
    with st.spinner("Extracting values from report..."):
        st.session_state.extracted_values = extract_values(llm, blood_report)
    with st.spinner("Generating health summary..."):
        st.session_state.summary = generate_summary(llm, st.session_state.extracted_values)
    with st.spinner("Generating diet plan..."):
        st.session_state.diet_plan = generate_diet_plan(llm, st.session_state.extracted_values)
    st.session_state.analyzed_report = blood_report


st.title("🩸 Blood Report Analysis")

uploaded_file = st.sidebar.file_uploader("Upload a blood report (.txt)", type=["txt"])
if uploaded_file is not None:
    blood_report = uploaded_file.read().decode("utf-8")
elif DEFAULT_REPORT_PATH.exists():
    blood_report = DEFAULT_REPORT_PATH.read_text(encoding="utf-8")
else:
    blood_report = ""

analyze_clicked = st.sidebar.button("Analyze Report", type="primary", disabled=not blood_report.strip())

if not os.environ.get("GOOGLE_API_KEY"):
    st.sidebar.error("GOOGLE_API_KEY is not set. Add it to your .env file.")

if analyze_clicked:
    run_analysis(blood_report)

left_col, summary_col, diet_col = st.columns([1.2, 1, 1])

with left_col:
    st.subheader("Blood Report")
    if blood_report.strip():
        st.text_area("Report contents", blood_report, height=600, label_visibility="collapsed")
    else:
        st.info("Upload a blood report to get started.")

with summary_col:
    st.subheader("Summary")
    if "summary" in st.session_state and st.session_state.get("analyzed_report") == blood_report:
        st.markdown(st.session_state.summary)
    else:
        st.info("Click **Analyze Report** to generate a summary.")

with diet_col:
    st.subheader("Diet Plan")
    if "diet_plan" in st.session_state and st.session_state.get("analyzed_report") == blood_report:
        st.markdown(st.session_state.diet_plan)
    else:
        st.info("Click **Analyze Report** to generate a diet plan.")

with st.expander("Extracted / classified values"):
    if "extracted_values" in st.session_state and st.session_state.get("analyzed_report") == blood_report:
        st.markdown(st.session_state.extracted_values)
    else:
        st.caption("Not generated yet.")
