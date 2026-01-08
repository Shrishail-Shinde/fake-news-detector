import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))



# Load datasets
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

# Add labels
true_df["label"] = 1   # Real news
fake_df["label"] = 0   # Fake news

# Combine datasets
df = pd.concat([true_df, fake_df], axis=0)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

#text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

#apply cleaning
df["clean_text"] = df["text"].apply(clean_text)

# Basic checks
print("Dataset shape:", df.shape)
print("\nSample rows:")
print(df.head())
print(df[["text", "clean_text"]].head())

# Save processed dataset
df.to_csv("data/processed_news.csv", index=False)

print("\nProcessed data saved to data/processed_news.csv")
