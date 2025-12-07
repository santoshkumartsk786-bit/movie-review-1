import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load your pre-trained model
model = load_model("lstm_imdb.h5")  # upload this to Colab first

# Helper functions
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    if len(encoded_review) == 0:
        encoded_review = [2]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment, float(prediction[0][0])

# Streamlit UI
st.title("IMDB Movie Review Sentiment Analysis -- Kumar")
st.write("Enter a movie review to classify it as positive or negative.")

review = st.text_area("Movie Review")

if st.button("Classify"):
    if review.strip() == "":
        st.write("Please enter a movie review.")
    else:
        sentiment, score = predict_sentiment(review)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score: {score:.4f}")
