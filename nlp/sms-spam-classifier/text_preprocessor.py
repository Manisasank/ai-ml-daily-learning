"""
Text Preprocessor & Feature Extraction Pipeline
Performs text normalization, regex entity replacement, stopword filtering,
and TF-IDF vectorization with n-gram support.
"""

import re
from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from .data_loader import load_data
except ImportError:
    from data_loader import load_data


# Lightweight set of English stopwords without requiring external NLTK downloads
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "would", "you", "your", "yours"
}


def clean_text(text: str) -> str:
    """
    Normalize text:
      - Replace URLs with 'token_url'
      - Replace phone numbers with 'token_phone'
      - Replace currency symbols and amounts with 'token_money'
      - Remove special characters and excess whitespace
      - Lowercase and filter common stop words
    """
    text = text.lower()
    
    # Normalize entities
    text = re.sub(r"https?://\S+|www\.\S+", " token_url ", text)
    text = re.sub(r"\b(?:\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b|\b\d{5,11}\b", " token_phone ", text)
    text = re.sub(r"[£$€]\s*\d+(?:,\d+)*(?:\.\d+)?|\b\d+\s*(?:dollars|pounds|cash)\b", " token_money ", text)
    
    # Strip punctuation and numbers
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    
    # Filter stopwords and short tokens
    filtered_tokens = [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]
    return " ".join(filtered_tokens)


def build_preprocessed_corpus(
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, TfidfVectorizer]:
    """
    Prepare cleaned corpus, split into train/test, and fit TF-IDF vectorizer.
    """
    df = load_data()
    df["cleaned_text"] = df["text"].apply(clean_text)
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"]
    )
    
    vectorizer = TfidfVectorizer(
        max_features=1500,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer


if __name__ == "__main__":
    sample = "WINNER! You have won £5000 cash. Visit http://promo.net or call 0871230022 immediately!"
    print(f"Original: {sample}")
    print(f"Cleaned : {clean_text(sample)}")
    
    X_tr, X_te, y_tr, y_te, vec = build_preprocessed_corpus()
    print(f"\nVocabulary size: {len(vec.vocabulary_)}")
    print(f"Train TF-IDF shape: {X_tr.shape}")
    print(f"Test TF-IDF shape: {X_te.shape}")
