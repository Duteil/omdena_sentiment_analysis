# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 19:45:24 2023

@author: Mathieu Duteil

Sentiment Analysis with Convolutional Neural Network (CNN)
This program reads a CSV file of preprocessed tweets, transforms 
the data, trains a CNN model, evaluates it, and saves the model.
"""

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump

class SentimentAnalysisModel:
    def __init__(self, max_features, maxlen):
        self.max_features = max_features
        self.maxlen = maxlen
        self.tokenizer = Tokenizer(num_words=self.max_features)
        self.model = self._build_model()
        
    def _build_model(self):
        # Build the model
        model = Sequential()

        # Start with an embedding layer
        model.add(Embedding(self.max_features, 50, input_length=self.maxlen))
        model.add(Dropout(0.2))

        # Add a Convolution1D
        model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
        model.add(GlobalMaxPooling1D())

        # Add a vanilla hidden layer
        model.add(Dense(250))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        
        return model

    def fit(self, X_train, y_train):
        # Tokenize and transform X_train to sequences of integers
        self.tokenizer.fit_on_texts(X_train)
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)

        # Pad the sequences
        X_train_vec = pad_sequences(X_train_seq, maxlen=self.maxlen)

        # Train the model
        self.model.fit(X_train_vec, y_train, epochs=5, batch_size=32, validation_split=0.1)

    def predict(self, X):
        # Transform X to sequences of integers using the trained tokenizer
        X_seq = self.tokenizer.texts_to_sequences(X)

        # Pad the sequences
        X_vec = pad_sequences(X_seq, maxlen=self.maxlen)

        # Predict
        return self.model.predict(X_vec)

    def evaluate(self, X_test, y_test):
        # Transform X_test to sequences of integers using the trained tokenizer
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)

        # Pad the sequences
        X_test_vec = pad_sequences(X_test_seq, maxlen=self.maxlen)

        # Evaluate
        return self.model.evaluate(X_test_vec, y_test)


# load and preprocess the dataset
dataset = pd.read_csv('preprocessed_tweets.csv')
X = dataset['text'].values
y = dataset['sentiment'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the vectorizer
vectorizer = CountVectorizer(max_features=3000)
vectorizer.fit(X_train)  # X_train is your preprocessed training data

# initialize and train the model
model = SentimentAnalysisModel(max_features=5000, maxlen=400)
model.fit(X_train, y_train)

# make predictions
predictions = model.predict(X_test)

# evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Save the model weights
model.model.save('Models/sentiment_model_CNN/sentiment_model_CNN.h5')

# Save the vectorizer
dump(vectorizer, 'Models/sentiment_model_CNN/vectorizer_cnn.joblib')