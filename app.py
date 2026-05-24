import importlib
import importlib.util
import joblib
import string
import nltk
from nltk.stem.porter import PorterStemmer

streamlit_spec = importlib.util.find_spec("streamlit")
if streamlit_spec is None:
    raise ModuleNotFoundError("The Streamlit package is required to run this app. Install it with `pip install streamlit`.")

st = importlib.import_module("streamlit")

nltk.download('stopwords')

stopwords = nltk.corpus.stopwords

# Load the model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return " ".join(words)

# Streamlit UI
st.title("📧 Email Spam Detection App")
st.write("Enter a message and check if it's spam or safe.")

user_input = st.text_area("✉️ Your Message:", height=150)

if st.button("Predict"):
    cleaned = preprocess_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    result = model.predict(vectorized)[0]

    if result == 1:
        st.error("🔴 Spam Message Detected!")
    else:
        st.success("🟢 This message is safe (Ham).")
