import joblib
import os

# Paths
MODEL_PATH = "models/fake_news_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

# Safety check
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model or vectorizer not found. Train the model first.")

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_news(text):
    """
    Predict whether a news article is Fake or Real.
    """
    text = str(text)
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]

    return "Real News" if prediction == 1 else "Fake News"


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    sample_news = input("Enter news text: ")
    result = predict_news(sample_news)
    print("\nPrediction:", result)
