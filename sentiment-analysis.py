import pandas as pd
import ast
import transformers
from transformers import pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained('bhadresh-savani/distilbert-base-uncased-emotion')

# last_hidden_states.numpy()
# print(last_hidden_states[0, 1].numpy())
    
subset_books_df = pd.read_csv('data/subset_books.csv')
classifier = pipeline("text-classification", model = 'bhadresh-savani/distilbert-base-uncased-emotion', top_k = 3)

for index, row in subset_books_df.head().iterrows():
    description = row['tokenized_description']  
    prediction = classifier(description)
    formatted_predictions = ", ".join([f"{entry['label']}: {entry['score']}" for entry in prediction[0]])
    subset_books_df.at[index, 'emotion_predictions'] = formatted_predictions

# print(subset_books_df.iloc[0, 3])
# for k in prediction[0]: print(k)

# book_tags_df = pd.read_csv("data/book_tags.csv")
books_df = pd.read_csv("data/books_1.Best_Books_Ever.csv")

stop_words = stopwords.words('english')
stop_words.extend(['from', 'story', 'novel', 'series', 'author', 'written', 'book', 'science_fiction', 'u', 'story', 'tale', 'writer', 'volume', 'classic',
                 'collection', 'el', 'novelette', 'shortstory', 'novella', 'story', 'wikipedia', 'essay'])
                
def preprocess_text(sen):
    if pd.isna(sen):
        return ""
    sentence = re.sub('[^a-zA-Z]', ' ', sen)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    tokens = word_tokenize(sentence.lower())
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def remove_stopwords(texts):
    return [[word for word in doc if word not in stop_words] for doc in texts]

def tokenize_description(description):
    tokens = preprocess_text(description)
    tokens = remove_stopwords([tokens])[0]
    return tokens

def extract_genres(genres):
    selected_genres = [genre for genre in genres if genre not in ["Fiction", "Classics", "Young Adult", "Adult"]] 
    return selected_genres[:2]

books_df['genres'] = books_df['genres'].apply(ast.literal_eval)
books_df['selected_genres'] = books_df['genres'].apply(extract_genres)
books_df['tokenized_description'] = books_df['description'].apply(tokenize_description)
selected_cols = ['title', 'selected_genres', 'tokenized_description', 'emotion_predictions']
books_df = books_df[books_df['language'].isin(['English'])]
subset_books_df = books_df.loc[:, selected_cols]
subset_books_df.to_csv("data/subset_books.csv")

