import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report


# -------------------------------------------------
# Load processed data
# -------------------------------------------------
df = pd.read_csv("data/processed_news.csv")

# Drop rows with missing cleaned text
df = df.dropna(subset=["clean_text"])

X = df["clean_text"].astype(str)
y = df["label"]


# -------------------------------------------------
# Train-test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------------------------------
# TF-IDF Vectorization
# -------------------------------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF Train shape:", X_train_tfidf.shape)
print("TF-IDF Test shape:", X_test_tfidf.shape)


# -------------------------------------------------
# Model 1: Logistic Regression
# -------------------------------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

lr_preds = lr_model.predict(X_test_tfidf)

print("\n===== Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, lr_preds))
print(classification_report(y_test, lr_preds))


# -------------------------------------------------
# Model 2: Naive Bayes
# -------------------------------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

nb_preds = nb_model.predict(X_test_tfidf)

print("\n===== Naive Bayes =====")
print("Accuracy:", accuracy_score(y_test, nb_preds))
print(classification_report(y_test, nb_preds))


# -------------------------------------------------
# Model 3: Decision Tree
# -------------------------------------------------
dt_model = DecisionTreeClassifier(
    max_depth=20,
    random_state=42
)
dt_model.fit(X_train_tfidf, y_train)

dt_preds = dt_model.predict(X_test_tfidf)

print("\n===== Decision Tree =====")
print("Accuracy:", accuracy_score(y_test, dt_preds))
print(classification_report(y_test, dt_preds))


# -------------------------------------------------
# Model 4: Random Forest
# -------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_tfidf, y_train)

rf_preds = rf_model.predict(X_test_tfidf)

print("\n===== Random Forest =====")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print(classification_report(y_test, rf_preds))


# -------------------------------------------------
# Save the BEST model (Logistic Regression)
# -------------------------------------------------
joblib.dump(lr_model, "models/fake_news_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n✅ Best model (Logistic Regression) and vectorizer saved successfully")

