from typing import Literal

import numpy as np
import streamlit as st
from keras.datasets import imdb  # type: ignore
from keras.models import load_model  # type: ignore
from keras.preprocessing import sequence  # type: ignore
from numpy.typing import NDArray

# Configure page
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load the RNN model (cached for efficiency)
@st.cache_resource
def load_rnn_model():
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


# Load model
model = load_rnn_model()


# Sentiment prediction function
def predict_sentiment(review: str) -> tuple[Literal["Positive", "Negative", "Error"], float]:
    if model is None:
        st.error("Model failed to load. Please check that 'models/imdb_rnn.keras' exists and is a valid Keras model.")
        return ("Error", 0.0)
    preprocessed_review = preprocess_input(review)
    prediction = model.predict(preprocessed_review, verbose=0)[0][0]
    return ("Positive" if prediction > 0.5 else "Negative", prediction)


# Theme Detection for Dark & Light Modes
theme_color = st.get_option("theme.primaryColor")
background_color = st.get_option("theme.backgroundColor")
text_color = st.get_option("theme.textColor")
border_color = theme_color if theme_color else "#2E86C1"

# UI Styling (Dynamic for Dark Mode)
st.markdown(
    f"""
    <style>
        .stTextArea textarea {{
            font-size: 16px;
            height: 200px !important;
        }}
        .stButton>button {{
            background-color: {border_color};
            color: white;
            border-radius: 8px;
            font-size: 18px;
            padding: 10px 20px;
            width: 100%;
        }}
        .prediction-container {{
            border: 2px solid {border_color};
            border-radius: 10px;
            background-color: {background_color};
            font-size: 20px;
            text-align: center;
            height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            padding: 20px;
            color: {text_color};
        }}
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
