"""
Author: Gnaneswari Tappetla
Title: implement Risk-Aware NLP Streamlit application with prediction and EDA modules
1. Added Streamlit UI with sidebar navigation (Prediction and EDA Analysis)
2. Integrated transformer-based risk classification model using HuggingFace
3. Implemented text preprocessing and weak labeling using regex-based risk/safe patterns
4. Added real-time prediction with probability metrics and risk level indicators
5. Included sample input buttons for quick testing scenarios
6. Implemented dataset upload functionality for exploratory data analysis (EDA)
7. Added visualizations: rating distribution, useful count histogram, top drugs bar chart
8. Added risk label distribution, review length boxplot, word count histogram
"""
import streamlit as st
 

st.set_page_config(
    page_title="Risk NLP System",
    layout="wide"
)

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

 
# CONFIG
 
MODEL_PATH = "/home/adarsh/Downloads/risk_transformer_model"
TEXT_COL = "review"
LABEL_COL = "risk_label"

RISK_PATTERNS = [
    r'double the dose', r'take more than prescribed', r'skip doctor',
    r'dont need doctor', r'don t need doctor', r'stop .* immediately',
    r'mix .* alcohol', r'share this medicine', r'works for everyone',
    r'increase dosage yourself', r'ignore side effects',
    r'no prescription needed', r'take two extra', r'self medicate'
]

SAFE_PATTERNS = [
    r'consult your doctor', r'ask your doctor', r'follow prescription',
    r'physician advised', r'doctor told me'
]

 
# LOAD MODEL
 
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

 
# TEXT CLEANING
 
def clean_text(x):
    x = str(x).lower()
    x = re.sub(r'<.*?>', ' ', x)
    x = re.sub(r'[^a-z0-9\s]', ' ', x)
    x = re.sub(r'\s+', ' ', x).strip()
    return x


def weak_label(text):
    t = clean_text(text)
    risk = sum(bool(re.search(p, t)) for p in RISK_PATTERNS)
    safe = sum(bool(re.search(p, t)) for p in SAFE_PATTERNS)
    return 1 if risk >= safe and risk > 0 else 0


 
# PREDICTION
 
def predict(text):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    unsafe_prob = probs[1]
    safe_prob = probs[0]

    label = "⚠️ UNSAFE" if unsafe_prob > 0.5 else "✅ SAFE"

    return label, safe_prob, unsafe_prob


 
# STREAMLIT UI
 
 

st.title("🧠 Risk-Aware NLP System ")

menu = st.sidebar.radio(
    "Navigation",
    ["🧾 Prediction", "📊 EDA Analysis"]
)

 
# PREDICTION PAGE
  
if menu == "🧾 Prediction":

    st.markdown("## Medical Risk Prediction System")
    st.write(
        "Enter a medical review / medicine advice text to detect whether it is Safe or Unsafe."
    )

        
    # SESSION STATE FOR TEXT
        
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

        
    # SAMPLE QUESTION BUTTONS
     
    st.markdown("### 💡 Try Sample Questions")

    col1, col2, col3 = st.columns(3)

     
    # COLUMN 1
     
    with col1:

        if st.button("💊 You can double the dose if pain continues"):
            st.session_state.input_text = \
                "You can double the dose if pain continues."

        if st.button("👨‍⚕️ Please consult your doctor before changing dose"):
            st.session_state.input_text = \
                "Please consult your doctor before changing dose."

        if st.button("🍷 Mix this medicine with alcohol for stronger effect."):
            st.session_state.input_text = \
                "Mix this medicine with alcohol for stronger effect."

        if st.button("💉 Increase the dosage yourself if symptoms continue."):
            st.session_state.input_text = \
                "Increase the dosage yourself if symptoms continue."


     
    # COLUMN 2
     
    with col2:

        if st.button("📋 Follow the prescription given by your doctor"):
            st.session_state.input_text = \
                "Follow the prescription given by your doctor."

        if st.button("⛔ Stop the medicine immediately without asking doctor"):
            st.session_state.input_text = \
                "Stop the medicine immediately without asking doctor."

        if st.button("🔥 Take two extra tablets if fever remains"):
            st.session_state.input_text = \
                "Take two extra tablets if fever remains."

        if st.button("🚫 Ignore side effects and continue taking medicine"):
            st.session_state.input_text = \
                "Ignore side effects and continue taking medicine."


     
    # COLUMN 3
     
    with col3:

        if st.button("🩺 Ask your physician before stopping this medicine"):
            st.session_state.input_text = \
                "Ask your physician before stopping this medicine."

        if st.button("🤝 Share this medicine with your family members"):
            st.session_state.input_text = \
                "Share this medicine with your family members."

       
        if st.button("✅ Use the medicine exactly as advised by your doctor"):
            st.session_state.input_text = \
                "Use the medicine exactly as advised by your doctor."
        
    # TEXT AREA
        
    text = st.text_area(
        "✍️ Input Text",
        key="input_text",
        height=180,
        placeholder="Type your own sentence or click sample buttons above..."
    )

        
    # PREDICT BUTTON
        
    predict_btn = st.button("🔍 Predict Risk")

    if predict_btn:

        if text.strip() == "":
            st.warning("⚠ Please enter some text.")
        else:

            with st.spinner("Analyzing text..."):
                label, safe_p, unsafe_p = predict(text)

            st.markdown("---")
            st.markdown("## 📌 Prediction Result")

            # RESULT
            if unsafe_p > 0.5:
                st.error(f"### {label}")
            else:
                st.success(f"### {label}")

            # METRICS
            c1, c2 = st.columns(2)

            c1.metric(
                "✅ Safe Probability",
                f"{safe_p*100:.2f}%"
            )

            c2.metric(
                "⚠ Unsafe Probability",
                f"{unsafe_p*100:.2f}%"
            )

            # PROGRESS
            st.markdown("### 📊 Confidence")

            st.write("Safe Confidence")
            st.progress(float(safe_p))

            st.write("Unsafe Confidence")
            st.progress(float(unsafe_p))

            # RISK LEVEL
            st.markdown("### 🚦 Risk Level")

            if unsafe_p < 0.30:
                st.success("🟢 Low Risk")
            elif unsafe_p < 0.60:
                st.warning("🟡 Moderate Risk")
            else:
                st.error("🔴 High Risk")
# EDA PAGE
elif menu == "📊 EDA Analysis":

    st.subheader("📂 Upload Dataset (drugsComTrain/Test CSV)")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:

        df = pd.read_csv(file)

          
        # DATA PREP
          
        df = df.dropna(subset=[TEXT_COL]).copy()
        df["clean_text"] = df[TEXT_COL].apply(clean_text)

        if LABEL_COL not in df.columns:
            df[LABEL_COL] = df["clean_text"].apply(weak_label)

        df["review_length"] = df["clean_text"].apply(len)
        df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))

          
        # KPI CARDS
          
        st.markdown("## 📌 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Rows", len(df))
        col2.metric("Columns", len(df.columns))
        col3.metric("Unsafe Reviews", int(df[LABEL_COL].sum()))
        col4.metric("Safe Reviews", int((df[LABEL_COL] == 0).sum()))

        st.dataframe(df.head())

        st.markdown("---")

          
        # GLOBAL STYLE
          
        sns.set_style("whitegrid")
        sns.set_context("talk")

          
        # RATING DISTRIBUTION
          
        if "rating" in df.columns:
            st.subheader("⭐ Rating Distribution")

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(
                x="rating",
                data=df,
                palette="viridis",
                edgecolor="black",
                ax=ax
            )

            ax.set_title("User Rating Frequency")
            st.pyplot(fig)

          
        # USEFUL COUNT
          
        if "usefulCount" in df.columns:
            st.subheader("👍 Useful Count Distribution")

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(
                df["usefulCount"],
                bins=50,
                kde=True,
                color="orange",
                ax=ax
            )

            ax.set_title("Helpful Votes Distribution")
            st.pyplot(fig)
