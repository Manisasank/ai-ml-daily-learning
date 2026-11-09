"""
NLP Model Training & Selection Module
Trains and compares Multinomial Naive Bayes, Complement Naive Bayes,
and Logistic Regression with 5-fold cross validation.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

try:
    from .text_preprocessor import build_preprocessed_corpus
except ImportError:
    from text_preprocessor import build_preprocessed_corpus


def train_and_compare_classifiers(X_train, y_train) -> Dict[str, Any]:
    """
    Train candidate text classification models and record cross-validation performance.
    """
    candidates = {
        "MultinomialNB": MultinomialNB(alpha=0.5),
        "ComplementNB": ComplementNB(alpha=0.5),
        "LogisticRegression": LogisticRegression(C=1.0, max_iter=500, random_state=42)
    }
    
    benchmark = {}
    
    for name, clf in candidates.items():
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="roc_auc")
        clf.fit(X_train, y_train)
        benchmark[name] = {
            "model": clf,
            "cv_roc_auc_mean": float(scores.mean()),
            "cv_roc_auc_std": float(scores.std())
        }
        print(f"[{name}] 5-Fold ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
    return benchmark


def get_top_predictive_words(vectorizer, model, top_n: int = 10) -> Tuple[List[str], List[str]]:
    """
    Extract highest probability tokens for Spam and Ham from Naive Bayes or Logistic Regression.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    if hasattr(model, "feature_log_prob_"):
        # Naive Bayes: diff between spam log-prob and ham log-prob
        spam_log_odds = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        top_spam_indices = np.argsort(spam_log_odds)[-top_n:][::-1]
        top_ham_indices = np.argsort(spam_log_odds)[:top_n]
    elif hasattr(model, "coef_"):
        # Logistic Regression coefficients
        coef = model.coef_[0]
        top_spam_indices = np.argsort(coef)[-top_n:][::-1]
        top_ham_indices = np.argsort(coef)[:top_n]
    else:
        return [], []
        
    top_spam = list(feature_names[top_spam_indices])
    top_ham = list(feature_names[top_ham_indices])
    return top_spam, top_ham


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, vectorizer = build_preprocessed_corpus()
    models = train_and_compare_classifiers(X_train, y_train)
    
    best_name = max(models.keys(), key=lambda k: models[k]["cv_roc_auc_mean"])
    best_clf = models[best_name]["model"]
    print(f"\nTop Performing Classifier: {best_name}")
    
    spam_words, ham_words = get_top_predictive_words(vectorizer, best_clf, top_n=8)
    print(f"\nTop Spam Keywords: {', '.join(spam_words)}")
    print(f"Top Ham Keywords : {', '.join(ham_words)}")
