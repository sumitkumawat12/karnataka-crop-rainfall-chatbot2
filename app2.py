import pandas as pd
import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt
import openpyxl

# ---------- Load dataset from GitHub ----------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/sumitkumawat12/karnataka-crop-rainfall-chatbot2/main/merge_data.xlsx"
    return pd.read_excel(url)

df = load_data()

st.title("🌾 Karnataka Crop & Rainfall Chatbot (Online)")
st.write("Ask questions about crop production and rainfall trends across Karnataka (2007–2016).")

if st.checkbox("Show dataset sample"):
    st.dataframe(df.head())

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def answer_query(question):
    q = question.lower()

    if "rainfall" in q and "year" in q:
        for year in df["Year"].unique():
            if str(year) in q:
                val = df[df["Year"] == year]["ANNUAL"].mean()
                return f"Average rainfall in {year} was {val:.2f} mm."
        return "Please specify a valid year for rainfall data."

    if "production" in q:
        for crop in df.columns:
            if "production" in crop.lower():
                crop_name = crop.split(" ")[0]
                if crop_name.lower() in q:
                    for year in df["Year"].unique():
                        if str(year) in q:
                            val = df[df["Year"] == year][crop].sum()
                            return f"{crop_name} production in {year} was {val:.2f} tonnes."
        return "I couldn't find that crop or year in the dataset."

    context = " ".join([f"{r.Year} {r.District} {r.SUBDIVISION}" for _, r in df.head(50).iterrows()])
    result = qa_pipeline(question=question, context=context)
    return result["answer"]

question = st.text_input("💬 Ask your question:")

if question:
    with st.spinner("Analyzing..."):
        try:
            response = answer_query(question)
            st.success(response)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

