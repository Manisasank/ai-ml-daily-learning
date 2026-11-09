"""
SMS Spam Dataset Generator & Loader
Generates a realistic synthetic corpus of SMS messages labeled as Ham (legitimate) or Spam.
"""

import os
from typing import Tuple
import pandas as pd


# Real-world message templates for robust synthetic generation
SPAM_TEMPLATES = [
    "WINNER! You have won a {prize} cash prize or a free holiday trip! Call {phone} now to claim.",
    "URGENT: Your bank account {account} has been flagged for suspicious activity. Verify immediately at {url}",
    "Congratulations! As a valued customer, you are pre-approved for a £{amount} personal loan with 0% APR. Reply YES.",
    "Free entry in 2 a weekly competition to win {prize}! Text WIN to {phone} to receive entry details. T&Cs apply.",
    "Exclusive Offer: Get 70% off your next {item} purchase today only! Click {url} before coupon expires.",
    "You have an unclaimed parcel at the depot. Redelivery fee £1.99 required. Visit {url} to reschedule delivery.",
    "ALERT: Your SIM card expires today. Top up immediately by texting TOPUP to {phone} to prevent service termination.",
    "Final reminder: You have {amount} unspent reward points expiring tonight! Redeem at {url} now."
]

HAM_TEMPLATES = [
    "Hey, are we still meeting for lunch at {time} tomorrow?",
    "Can you please send me the lecture notes from today's machine learning class?",
    "I'm running about 10 minutes late, sorry! Traffic is pretty heavy on the highway.",
    "Mom asked if you can pick up some milk and bread on your way home.",
    "Did you see the latest update on the project repository? Looks great!",
    "Happy Birthday! Hope you have a wonderful day and a great year ahead.",
    "Thanks for helping me with the code review earlier, really appreciated it.",
    "Let me know when you reach home safely. See you on Monday!",
    "Don't forget we have our weekly team sync at {time}. Meeting link is in your calendar.",
    "Just finished reading the paper on transformer architectures, let's discuss it later."
]


def generate_sms_dataset(n_samples: int = 1500, spam_ratio: float = 0.20, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic labeled SMS messages.
    
    Parameters:
        n_samples: Total number of records to generate.
        spam_ratio: Proportion of messages that are spam.
        random_state: Seed for reproducibility.
        
    Returns:
        pd.DataFrame with 'text' and 'label' columns (0 = ham, 1 = spam).
    """
    import random
    random.seed(random_state)
    
    prizes = ["$1,000,000", "$5,000", "iPhone 15 Pro", "BMW 3-Series", "Luxury Cruise"]
    urls = ["http://bit.ly/claim-win", "http://secure-verify.co/login", "http://parcel-track24.com", "http://promo-deal77.net"]
    phones = ["0871230022", "09061743811", "88066", "82242"]
    items = ["designer shoes", "electronics", "smart watch", "laptop"]
    times = ["1:00 PM", "2:30 PM", "11:00 AM", "6:00 PM", "dinner time"]
    
    n_spam = int(n_samples * spam_ratio)
    n_ham = n_samples - n_spam
    
    texts = []
    labels = []
    
    # Generate Spam
    for _ in range(n_spam):
        template = random.choice(SPAM_TEMPLATES)
        text = template.format(
            prize=random.choice(prizes),
            url=random.choice(urls),
            phone=random.choice(phones),
            amount=random.randint(500, 10000),
            account=f"XX{random.randint(1000, 9999)}",
            item=random.choice(items)
        )
        texts.append(text)
        labels.append(1)
        
    # Generate Ham
    for _ in range(n_ham):
        template = random.choice(HAM_TEMPLATES)
        text = template.format(
            time=random.choice(times)
        )
        texts.append(text)
        labels.append(0)
        
    # Shuffle
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    
    df = pd.DataFrame({"text": list(texts), "label": list(labels)})
    return df


def load_data(save_path: str = None) -> pd.DataFrame:
    """
    Load existing CSV or generate dataset and optionally cache it.
    """
    if save_path and os.path.exists(save_path):
        return pd.read_csv(save_path)
    
    df = generate_sms_dataset(n_samples=1600, spam_ratio=0.22, random_state=42)
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Generated {len(df)} SMS messages.")
    print(f"Spam count: {(df['label'] == 1).sum()} ({(df['label'] == 1).mean():.1%})")
    print(f"Ham count: {(df['label'] == 0).sum()} ({(df['label'] == 0).mean():.1%})")
    print("\nSample records:")
    for _, row in df.head(3).iterrows():
        label_str = "SPAM" if row['label'] == 1 else "HAM"
        print(f"[{label_str}] {row['text']}")
