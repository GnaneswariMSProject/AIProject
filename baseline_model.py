# src/baseline_model.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def train_baseline_model(df):

    X = df["clean_review"]
    y = df["label"]

    # Split training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    # Logistic Regression
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"  # Important for unsafe recall
    )

    model.fit(X_train_tfidf, y_train)

    # Predictions
    y_pred = model.predict(X_val_tfidf)

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    return model, vectorizer