"""
Evaluation Suite & Real-Time SMS Spam Inference CLI
Calculates rigorous classification metrics on held-out test data
and provides an interactive inference engine for raw text messages.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

try:
    from .text_preprocessor import clean_text, build_preprocessed_corpus
    from .model_trainer import train_and_compare_classifiers
except ImportError:
    from text_preprocessor import clean_text, build_preprocessed_corpus
    from model_trainer import train_and_compare_classifiers


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


def classify_message(raw_text: str, model, vectorizer) -> Dict[str, Any]:
    """
    Predict spam probability and category for an arbitrary raw SMS string.
    """
    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])
    
    if hasattr(model, "predict_proba"):
        prob_spam = float(model.predict_proba(vec)[0, 1])
    else:
        prob_spam = float(model.predict(vec)[0])
        
    is_spam = prob_spam >= 0.50
    confidence = "Very High" if prob_spam > 0.85 or prob_spam < 0.15 else "Moderate"
    
    return {
        "text": raw_text,
        "cleaned_text": cleaned,
        "prediction": "SPAM" if is_spam else "HAM",
        "spam_probability": round(prob_spam, 4),
        "confidence": confidence
    }


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, vectorizer = build_preprocessed_corpus()
    models = train_and_compare_classifiers(X_train, y_train)
    clf = models["MultinomialNB"]["model"]
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Test Performance Report ---")
    print(classification_report(y_test, y_pred, target_names=["HAM", "SPAM"]))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"[[TN={cm[0,0]}  FP={cm[0,1]}],")
    print(f" [FN={cm[1,0]}  TP={cm[1,1]}]]")
    
    # Test on unseen raw test strings
    test_messages = [
        "Hey bro, do you want to play badminton this evening at 6?",
        "URGENT! You have won $10,000 cash prize! Text CLAIM to 88066 now to receive voucher.",
        "Can you review the pull request before our deployment call?",
        "Congratulations! 80% discount on summer sale, click http://tinyurl.com/deal-now"
    ]
    
    print("\n--- Live Inference Engine Demo ---")
    for msg in test_messages:
        res = classify_message(msg, clf, vectorizer)
        print(f"\nMessage: '{res['text']}'")
        print(f"Result : [{res['prediction']}] (Spam Probability: {res['spam_probability']:.2%}, Confidence: {res['confidence']})")
