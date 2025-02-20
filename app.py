from typing import Literal

import numpy as np
import streamlit as st
from keras.api.datasets import imdb
from keras.api.models import Sequential, load_model
from keras.api.preprocessing import sequence
from numpy.typing import NDArray

# Configure page
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load the RNN model (cached for efficiency)
@st.cache_resource
def load_rnn_model() -> Sequential:
    return load_model("models/imdb_rnn.keras")


# Load word index once and cache it
@st.cache_resource
def load_word_index() -> dict[str, int]:
    word_index: dict[str, int] = imdb.get_word_index()
    return {word: (index + 3) for word, index in word_index.items()}


# Preprocess user input efficiently
@st.cache_data
def preprocess_input(text: str) -> NDArray[np.int64]:
    word_index = load_word_index()
    encoded_input: NDArray[np.int64] = np.array(
        [[word_index.get(word, 2) for word in text.lower().split()]], dtype=np.int64
    )
    return sequence.pad_sequences(encoded_input, maxlen=500)


# Sentiment prediction function
def predict_sentiment(review: str) -> tuple[Literal["Positive", "Negative"], float]:
    preprocessed_review = preprocess_input(review)
    prediction = model.predict(preprocessed_review, verbose=0)[0][0]
    return ("Positive" if prediction > 0.5 else "Negative", prediction)


# Load model
model = load_rnn_model()

# UI Styling
st.markdown(
    """
    <style>
        .stTextArea textarea {
            font-size: 16px;
            height: 150px !important;
        }
        .stButton>button {
            background-color: #2E86C1;
            color: white;
            border-radius: 8px;
            font-size: 18px;
            padding: 10px 20px;
            width: 100%;
        }
        .prediction-container {
            border: 2px solid #2E86C1;
            border-radius: 10px;
            background-color: #F0F8FF;
            font-size: 20px;
            text-align: center;
            height: 150px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            padding: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>IMDB Movie Review Sentiment Analysis</h1>",
    unsafe_allow_html=True,
)
st.write("Enter a movie review below to predict the sentiment.")

# Layout with columns: Text Input (Left), Prediction Output (Right)
col1, col2 = st.columns([2, 1])

with col1:
    user_review = st.text_area(
        "Enter your review",
        placeholder="Type your review here...",
        key="review_input",
        label_visibility="collapsed",
    )

with col2:
    # st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing to align with text box
    if "sentiment" in st.session_state:
        st.markdown(
            f"<div class='prediction-container'>"
            f"<b>Sentiment:</b> {st.session_state['sentiment']} <br> "
            f"<b>Confidence:</b> {st.session_state['confidence']:.4f}"
            f"</div>",
            unsafe_allow_html=True,
        )

# Centered Predict button BELOW the columns
predict_button = st.button("Predict Sentiment")

if predict_button:
    with st.spinner("Analyzing sentiment..."):
        sentiment, confidence = predict_sentiment(user_review)
        st.session_state["sentiment"] = sentiment
        st.session_state["confidence"] = confidence
        st.rerun()

# Footer
st.markdown(
    "<p style='text-align: center; color: gray;'>Powered by Keras and Streamlit</p>",
    unsafe_allow_html=True,
)
