# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 19:45:24 2023

@author: Mathieu Duteil

This script trains an LSTM model for sentiment analysis
on preprocessed tweets.

"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_twitter_data(path):
    """
    Loads the preprocessed tweet data, splits it into training, validation and test sets,
    and returns these along with the text tokenizer.
    
    Returns:
    X_train, y_train, X_val, y_val, X_test, y_test, encoder: Features and targets for training, validation, 
    and testing, and the trained text tokenizer.
    """
    
    # Load the data
    df = pd.read_csv(path)

    # Convert 'tokens' column to string
    df['tokens'] = df['tokens'].astype(str)

    # Split the data into features (X) and target (y)
    X = df['tokens']
    y = df['sentiment']

    # Tokenize the text
    encoder = Tokenizer()
    encoder.fit_on_texts(X)

    # Convert text to sequences of integers
    X_seq = encoder.texts_to_sequences(X)

    # Pad sequences to the same length
    X_pad = pad_sequences(X_seq, maxlen=128)

    # Split the data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test, encoder


# Define the LSTM model
def LSTM_Model(encoder):
    """
    Defines the LSTM model for sentiment analysis.
    
    Args:
    encoder: The text tokenizer trained on the tweet data.
    
    Returns:
    model: The LSTM model.
    """
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=len(encoder.word_index) + 1, output_dim=64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])
    return model

if __name__ == "__main__":
    # Load the data
    X_train, y_train, X_val, y_val, X_test, y_test, encoder = get_twitter_data(path = 'preprocessed_tweets.csv')

    # Load the model weights if they exist
    if os.path.exists("twitter_weights"):
        sentiment_model = tf.keras.models.load_model("twitter_weights")
    else:
        sentiment_model = LSTM_Model(encoder)

    # Compile the model
    sentiment_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                            optimizer=tf.keras.optimizers.Adam(1e-4),
                            metrics=['accuracy'])

    # Train the model
    history = sentiment_model.fit(X_train, y_train, epochs=10, batch_size=1024,
                                  validation_data=(X_val, y_val))

    # Evaluate the model
    test_loss, test_acc = sentiment_model.evaluate(X_test, y_test)
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)

    # Save the model weights
    sentiment_model.save("twitter_weights_LSTM")

