"""
Created on Wed Jul 19 19:45:24 2023

@author: Mathieu Duteil

This script trains a LinearSVC model for sentiment analysis on 
preprocessed tweets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from joblib import dump

def get_twitter_data(path):
    """
    Loads the preprocessed tweet data, splits it into training and testing sets,
    and vectorizes the tweet text.
    
    Returns:
    X_train, y_train, X_test, y_test, vectorizer: Features and targets for training and testing, 
    and the trained CountVectorizer.
    """
    
    # Load the data
    df = pd.read_csv(path)

    # Split the data into features (X) and target (y)
    X = df['tokens']
    y = df['sentiment']

    # Vectorize the text
    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test, vectorizer


if __name__ == "__main__":
    # Load the data
    X_train, y_train, X_test, y_test, vectorizer = get_twitter_data(path = "preprocessed_tweets.csv")

    # Initialize and train the model
    model = LinearSVC()
    model.fit(X_train, y_train)

    # Predict the test set results
    y_pred = model.predict(X_test)

    # Print the accuracy of the model
    print('Test Accuracy:', accuracy_score(y_test, y_pred))
    
    # Save the model and vectorizer
    dump(model, 'sentiment_svc.joblib')
    dump(vectorizer, 'vectorizer.joblib')
