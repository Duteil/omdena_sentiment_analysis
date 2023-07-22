# Import necessary libraries
from tensorflow.keras.models import load_model
from sklearn.svm import LinearSVC
from transformers.models.distilbert import TFDistilBertForSequenceClassification
import pickle
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
    # Load the saved vectorizer from a file
    if model_name == "LinearSVC":
        with open("./Models/sentiment_model_linearSVC/vectorizer.joblib", "rb") as f:
            vectorizer = load(f)
    elif model_name in ["CNN", "LSTM"]:
        with open("./Models/sentiment_model_" + model_name + "/vectorizer_cnn.joblib", "rb") as f:
            vectorizer = load(f)
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
        sentiment_model = load("./Models/sentiment_model_linearSVC/sentiment_svc.joblib")
    elif model_name == "CNN":
        sentiment_model = load_model("./Models/sentiment_CNN.h5", compile=False)
    elif model_name == "LSTM":
        sentiment_model = load_model("./Models/sentiment_LSTM.h5", compile=False)
    else:  # model_name == "Transformer"
        sentiment_model = TFDistilBertForSequenceClassification.from_pretrained("./Models/sentiment_model_bert/")
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
