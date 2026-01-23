import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

def predict_news(text):
    text = clean_text(str(text))   # same cleaning as training
    text_tfidf = vectorizer.transform([text])

    proba = model.predict_proba(text_tfidf)[0]
    print("Prob Fake:", proba[0], "Prob Real:", proba[1])

    prediction = model.predict(text_tfidf)[0]
    return "Real News" if prediction == 1 else "Fake News"