# src/label_strategy.py

def create_unsafe_label(df):
    """
    Create initial binary labels:
    1 = Unsafe
    0 = Safe

    This is a rule-based starting point.
    Later you should manually verify.
    """

    unsafe_keywords = [
        "double dose",
        "increase dose",
        "skip doctor",
        "without doctor",
        "mixed with alcohol",
        "took more than",
        "overdose"
    ]

    def check_unsafe(text):
        for keyword in unsafe_keywords:
            if keyword in text:
                return 1
        return 0

    df["label"] = df["clean_review"].apply(check_unsafe)

    print("Label distribution:")
    print(df["label"].value_counts())

    return df