# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 20:20:55 2023

@author: Mathieu Duteil


This script preprocesses a CSV file of tweets for sentiment analysis. 
It loads the file, applies text preprocessing (such as removing URLs, 
mentions, hashtags, special characters, punctuation and stopwords), 
performs tokenization, POS tagging, and lemmatization, and saves the 
processed data into a new CSV file.

"""

from sklearn.preprocessing import LabelEncoder
import nltk
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from gensim.parsing.preprocessing import STOPWORDS
import pandas as pd
from tqdm import tqdm
import re

def nltk_to_wordnet_pos(nltk_pos):
    """
    Converts nltk pos tags to wordnet pos tags.
    
    Args:
    nltk_pos (str): The pos tag from nltk.pos_tag.
    
    Returns:
    str: The corresponding wordnet pos tag.
    """
    if nltk_pos.startswith('J'):
        return wordnet.ADJ
    elif nltk_pos.startswith('V'):
        return wordnet.VERB
    elif nltk_pos.startswith('N'):
        return wordnet.NOUN
    elif nltk_pos.startswith('R'):
        return wordnet.ADV
    else: 
        return None

def preprocess_text(text):
    """
    Preprocesses a string of text by removing URLs, mentions, hashtags, 
    special characters, punctuation, stopwords, performs tokenization, 
    POS tagging, and lemmatization.
    
    Args:
    text (str): The input text.

    Returns:
    list: A list of tuples where each tuple contains the preprocessed 
    proposition and the original proposition.
    """
    
    # Check if the input is a string
    if not isinstance(text, str):
        return []
    
    # Instantiating our lemmatizer
    lemma = WordNetLemmatizer()
    
    # Ensure the text ends with a punctuation mark
    if not re.search(r"[,;:.?!]$", text):
        text += "."
    
    # Splitting the text along relevant punctuation
    split_propositions = re.split(r'([,;:.?!])', text)
    
    # Corrected pairing of propositions with punctuation
    if len(split_propositions) > 1:
        split_propositions = [prop + punct for prop, punct in zip(split_propositions[::2], split_propositions[1::2])]
    else:
        split_propositions = [text]
    
    processed_propositions = []
    
    for proposition in split_propositions:
        # Remove URLs, mentions, hashtags, special characters, and punctuation
        tokens = ' '.join(word for word in proposition.split() if not word.startswith(('http', 'www')))
        tokens = ' '.join(word for word in tokens.split() if not '.ly' in word)
        tokens = ' '.join(word for word in tokens.split() if not '.co' in word)
        tokens = ' '.join(word for word in tokens.split() if not word.startswith('@'))
        tokens = ' '.join(word[1:] if word.startswith('#') else word for word in tokens.split())
        tokens = ''.join(char for char in tokens if char.isalnum() or char.isspace())
        
        # Removing stopwords
        stopwords = set(STOPWORDS)
        stopwords.remove("not")
        tokens = ' '.join(word for word in tokens.split() if word not in stopwords)
    
        # Tokenization
        tokens = word_tokenize(tokens)
        
        # POS tagging
        pos_tags = pos_tag(tokens)        
    
        # Lemmatization
        tokens = ' '.join([lemma.lemmatize(word, nltk_to_wordnet_pos(pos)) if nltk_to_wordnet_pos(pos) else word for word, pos in pos_tags])
        
        if tokens != '':
            processed_propositions.append((tokens, proposition))
            
    return processed_propositions

def preprocess_tweets(file_path):
    """
    Preprocesses a CSV file of tweets for sentiment analysis. 
    It loads the file, applies text preprocessing (using preprocess_text 
    function), and saves the processed data into a new CSV file.
    
    Args:
    file_path (str): The path to the CSV file.

    Returns:
    None
    """
    
    # Initialize the LabelEncoder
    le = LabelEncoder()
    
    # Loading the dataset
    if 'sentiment140' in file_path:
        df = pd.read_csv(file_path, names=['target', 'ids', 'date', 'flag', 'user', 'text'], encoding='latin-1')
        df.rename(columns={'target': 'sentiment'}, inplace=True)
        
        # Change sentiment values from 0, 2, 4 to 'negative', 'neutral', 'positive'
        sentiment_mapping = {0: 'negative', 2: 'neutral', 4: 'positive'}
        df['sentiment'] = df['sentiment'].map(sentiment_mapping)
    else:
        df = pd.read_csv(file_path, encoding='latin-1')
        df = df[['sentiment', 'text']]
        
    # Ensure sentiment values are 'negative', 'neutral', 'positive'
    df['sentiment'] = le.fit_transform(df['sentiment'])

    # Drop rows with non-string text or non [0,1,2] sentiment
    df = df[df['sentiment'].isin([0, 1, 2])]
    df = df[df['text'].apply(lambda x: isinstance(x, str))]

    # Apply the preprocess_text function to create the "tokens" column
    tqdm.pandas(desc="Processing tweets")
    df['processed'] = df['text'].progress_apply(preprocess_text)
    
    # Drop the rows where 'processed' is an empty list
    df = df[df['processed'].str.len() != 0]
    
    tqdm.pandas(desc="Processing tweets")
    df['tokens'] = df['processed'].progress_apply(lambda x: ' '.join([item[0] for item in x]))

    # Save the processed data
    df.to_csv('preprocessed_tweets.csv', index=False)
