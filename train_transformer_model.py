"""
Sentiment Analysis with Transformer (DistilBert) Model
This program reads a CSV file of preprocessed tweets, transforms the data, trains a transformer model, evaluates it, and saves the model.
"""

import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

def convert_df_to_tf_data(df, text_col, label_col, max_length=128):
    """
    Function to convert pandas dataframe to tf.data.Dataset.

    Parameters:
    df: DataFrame, input data.
    text_col: str, name of the column in df containing the text.
    label_col: str, name of the column in df containing the label.
    max_length: int, maximum length of the text (default is 128).

    Returns:
    tf.data.Dataset object.
    """
    input_ids = []
    attention_masks = []
    labels = df[label_col].values

    for text in tqdm(df[text_col].values):
        encoded_dict = tokenizer.encode_plus(
            text,                      
            add_special_tokens = True, 
            max_length = max_length,  
            pad_to_max_length = True,
            return_attention_mask = True,   
            return_tensors = 'tf',     
        )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = tf.concat(input_ids, axis=0)
    attention_masks = tf.concat(attention_masks, axis=0)
    labels = tf.convert_to_tensor(labels)

    return tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': attention_masks}, labels))

df = pd.read_csv('small_preprocessed_tweets.csv')
df['tokens'] = df['tokens'].astype(str)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

print("Loading...")
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

print("Converting train dataframe to tf.data.Dataset...")
train_data = convert_df_to_tf_data(train_df, 'tokens', 'sentiment')
print("Converting validation dataframe to tf.data.Dataset...")
val_data = convert_df_to_tf_data(val_df, 'tokens', 'sentiment')
print("Converting test dataframe to tf.data.Dataset...")
test_data = convert_df_to_tf_data(test_df, 'tokens', 'sentiment')

train_data = train_data.shuffle(100).batch(16).repeat(2)
val_data = val_data.batch(16)

print("Compiling the model.")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-08, clipnorm=1.0), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

print("Training started.")
model.fit(train_data, epochs=2, validation_data=val_data, verbose=1, callbacks=[early_stopping])

test_loss, test_acc = model.evaluate(test_data.batch(16))
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

model.save_pretrained("./Models/sentiment_model_bert")
