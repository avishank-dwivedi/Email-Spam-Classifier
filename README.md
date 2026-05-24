
###https://email-spam-classifier-43nr.onrender.com/



# 📧 Spam Message Classifier using NLP and Machine Learning

This project is an end-to-end machine learning solution that classifies SMS or email text as **Spam** or **Not Spam (Ham)**. It uses **Natural Language Processing (NLP)** techniques and a **Naive Bayes classifier** trained on the SMS Spam Collection Dataset.

---

## 🚀 Features

- Cleaned and preprocessed SMS text
- TF-IDF feature extraction
- Trained with Naive Bayes Classifier
- Achieves 95–98% accuracy
- Live prediction with Streamlit web app

---

## 🧠 Technologies Used

- Python
- NLTK (Natural Language Toolkit)
- Scikit-learn
- Pandas
- Streamlit

---

## 📁 Dataset

We used the public [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), which contains 5,572 labeled messages:

| Label | Message |
|-------|---------|
| ham   | Hello, how are you? |
| spam  | Win cash now!!! Click here |

---

## 🧹 Data Preprocessing

Steps:
- Lowercasing
- Removing punctuation
- Tokenization
- Stopword removal
- Stemming (Porter Stemmer)
- TF-IDF Vectorization

---

## 🤖 Model Training

We used **Multinomial Naive Bayes**, which works well for text classification problems.

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)

