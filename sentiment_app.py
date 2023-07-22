# Import necessary libraries
from tensorflow.keras.models import load_model
from sklearn.svm import LinearSVC
from transformers.models.distilbert import TFDistilBertForSequenceClassification
import pickle
import os
from joblib import load
import numpy as np
import streamlit as st
from my_preprocessor import preprocess_text

# Building the front end
st.title("Sentiment Analysis App")

st.markdown("""
This app attempts to identify which parts of a text are positive or negative. 
Please write some text in the field for the model to help you analyse its different parts.
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

@st.cache_resource
def vec(model_name):
    if model_name == "LinearSVC":
        vectorizer_path = "./Models/sentiment_model_linearSVC/vectorizer.joblib"
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, "rb") as f:
                vectorizer = load(f)
        else:
            vectorizer = None
    elif model_name in ["CNN", "LSTM"]:
        vectorizer_path = "./Models/sentiment_model_" + model_name + "/vectorizer_cnn.joblib"
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, "rb") as f:
                vectorizer = load(f)
        else:
            vectorizer = None
    elif model_name == "Transformer":
        vectorizer = None  # Transformers use their own tokenization, so no vectorizer is needed
    return vectorizer


# Display model selection box
st.sidebar.title("Model Selection")
model_name = st.sidebar.selectbox("Choose a model", ["LinearSVC", "CNN", "LSTM", "Transformer"])
text_vectorizer = vec(model_name)

@st.cache_resource
def get_model(model_name):
    if model_name == "LinearSVC":
        model_path = "./Models/sentiment_model_linearSVC/sentiment_svc.joblib"
        if os.path.exists(model_path):
            sentiment_model = load(model_path)
        else:
            sentiment_model = None
    elif model_name == "CNN":
        model_path = "./Models/sentiment_model_CNN/sentiment_CNN.h5"
        if os.path.exists(model_path):
            sentiment_model = load_model(model_path, compile=False)
        else:
            sentiment_model = None
    elif model_name == "LSTM":
        model_path = "./Models/sentiment_model_LSTM/sentiment_LSTM.h5"
        if os.path.exists(model_path):
            sentiment_model = load_model(model_path, compile=False)
        else:
            sentiment_model = None
    else:  # model_name == "Transformer"
        model_path = "./Models/sentiment_model_bert/tf_model.h5"
        if os.path.exists(model_path):
            sentiment_model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
        else:
            sentiment_model = None

    return sentiment_model

def predict_and_color_text(text, model_name):
    # Preprocess the text and split it into grammatical propositions
    processed_text = preprocess_text(text)
    
    # Initialize an empty string to store the colored text
    colored_text = ""
    
    # Iterate through each proposition
    for token_prop, orig_prop in processed_text:
        # Vectorize the tokenized proposition
        vectorized_prop = text_vectorizer.transform([token_prop])
        
        # Predict the sentiment of the proposition
        prediction = sentiment_model.predict(vectorized_prop)
        sentiment = np.round(prediction[0])

        # Color the proposition based on its sentiment
        if sentiment == 0:  # Negative
            colored_prop = f'<span style="color:red">{orig_prop}</span>'
        elif sentiment == 1:  # Neutral
            colored_prop = orig_prop  # Let's leave the neutral sentiment as black (default text color)
        else:  # Positive
            colored_prop = f'<span style="color:green">{orig_prop}</span>'
        
        colored_text += " " + colored_prop
    
    return colored_text


# Load the selected model
sentiment_model = get_model(model_name)

# Create a text area for user input
text = st.text_area("Enter Text:")

# When the user enters text, color it and display it
if text:
    colored_text = predict_and_color_text(text, model_name)
    st.markdown(colored_text, unsafe_allow_html=True)
