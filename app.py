from flask import Flask, render_template, request
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# -------------------------------------------------
# Load trained model and vectorizer
# -------------------------------------------------
MODEL_PATH = "models/fake_news_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

# Safety check
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model or vectorizer not found. Please train the model first.")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# -------------------------------------------------
# Prediction function
# -------------------------------------------------
def predict_news(text):
    text = str(text)
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return "Real News" if prediction == 1 else "Fake News"


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    news_text = ""

    if request.method == "POST":
        news_text = request.form.get("news")
        result = predict_news(news_text)

    return render_template(
        "index.html",
        result=result,
        news_text=news_text
    )


# -------------------------------------------------
# Run app
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
