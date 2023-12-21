import pandas as pd
import transformers
from transformers import pipeline
from emotion_mapping import emotion_synonyms, expanded_emotion_synonyms

classifier = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)

def get_emotion_predictions(text):
    for emotion, expanded_emotion_synonyms in emotion_synonyms.items():
        if any(word in text.lower() for word in expanded_emotion_synonyms):
            return emotion
    prediction = classifier(text)
    sorted_predictions = sorted(prediction[0], key=lambda x: x['score'], reverse=True)
    top_predictions = sorted_predictions[:3]
    formatted_predictions = ", ".join([f"{entry['label']}: {entry['score']}" for entry in top_predictions])
    return formatted_predictions

def recommend_book(prompt):
    user_input = input(prompt)
    emotion_predictions = get_emotion_predictions(user_input)
    print("top emotions:", emotion_predictions)

prompt1 = "give me a book that feels like: "
prompt2 = "give me a book that makes me: "

# user_input = input("enter a sentence: ")
# emotion_predictions = get_emotion_predictions(user_input)
# print("top emotions:", emotion_predictions)

recommend_book(prompt2)
recommend_book(prompt1)