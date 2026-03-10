import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/val.csv"
TEST_PATH = "data/processed/test.csv"


def load_split(path: str):
    df = pd.read_csv(path).dropna(subset=["text", "label"])
    return df["text"].tolist(), df["label"].tolist()


def evaluate(name: str, y_true, y_pred) -> None:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    print(f"\n{name} Results")
    print("-" * 40)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def main() -> None:
    os.makedirs("results", exist_ok=True)

    X_train, y_train = load_split(TRAIN_PATH)
    X_val, y_val = load_split(VAL_PATH)
    X_test, y_test = load_split(TEST_PATH)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    clf.fit(X_train_tfidf, y_train)

    val_preds = clf.predict(X_val_tfidf)
    test_preds = clf.predict(X_test_tfidf)

    evaluate("Validation", y_val, val_preds)
    evaluate("Test", y_test, test_preds)


if __name__ == "__main__":
    main()