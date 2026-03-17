# src/preprocess.py

import pandas as pd
import re


def load_data(filepath):
    """
    Load the Drugs.com dataset
    """
    df = pd.read_csv(filepath)
    print("Dataset loaded successfully.")
    print("Shape:", df.shape)
    return df


def clean_text(text):
    """
    Basic cleaning:
    - Lowercase
    - Remove special characters
    - Remove extra spaces
    """
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_reviews(df):
    """
    Clean review column
    """
    df = df.dropna(subset=["review"])
    df["clean_review"] = df["review"].apply(clean_text)
    return df