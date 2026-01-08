# 📰 Fake News Detection using Machine Learning

An end-to-end **Fake News Detection system** that classifies news articles as **Real** or **Fake** using **Natural Language Processing (NLP)** and **Machine Learning**.  
The project includes data preprocessing, model training, evaluation, and a **Flask-based web application** for real-time prediction.

---

## 🚀 Features

- Text preprocessing and cleaning using NLP techniques
- TF-IDF feature extraction (unigrams + bigrams)
- Comparison of multiple ML models
- Overfitting analysis and model selection
- Final deployment using Flask with HTML & CSS UI
- Real-time fake/real news prediction

---

## 🧠 Machine Learning Models Used

| Model | Purpose |
|------|--------|
| Logistic Regression | **Final deployed model** (best generalization) |
| Naive Bayes | Baseline comparison |
| Decision Tree | Overfitting analysis |
| Random Forest | Ensemble comparison |

> **Logistic Regression** was selected as the final model due to superior performance on high-dimensional TF-IDF features.

---

## 📊 Dataset

- **True.csv** – Real news articles  
- **Fake.csv** – Fake news articles  

**Total records:** ~42,000+  
Each article includes:
- `text`
- `subject`
- `date`

A cleaned dataset is saved as: `data/processed_news.csv`


---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:**  
  - Pandas, NumPy  
  - NLTK  
  - Scikit-learn  
  - Joblib  
- **Backend:** Flask  
- **Frontend:** HTML, CSS  

---

## 📁 Project Structure

```bash
fake-news-detection/
│
├── app.py
│
├── data/
│ ├── True.csv
│ ├── Fake.csv
│ └── processed_news.csv
│
├── models/
│ ├── fake_news_model.pkl
│ └── vectorizer.pkl
│
├── src/
│ ├── data_preprocessing.py
│ ├── train_model.py
│ └── predict.py
│
├── templates/
│ └── index.html
│
├── static/
│ └── style.css
│
├── requirements.txt
└── README.md
```


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone 
cd fake-news-detection
```

### Create and activate virtual environment

```bash
python -m venv venv
```

- Activate the virtual environment
```bash
venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### 🔄 Run the Project
- Step 1: Preprocess Data
python src/data_preprocessing.py

- Step 2: Train Models
python src/train_model.py

- Step 3: Run Flask Web App
python app.py


Open browser: http://127.0.0.1:5000/

🖥️ Web Application

User pastes a news article

Clicks Check News

Model predicts:

✅ Real News

❌ Fake News

The UI is built using HTML & CSS and connected to the ML model via Flask.

### 📈 Results

- Logistic Regression Accuracy: ~98%

- Tree-based models showed overfitting due to sparse TF-IDF features

- Logistic Regression generalized best and was deployed