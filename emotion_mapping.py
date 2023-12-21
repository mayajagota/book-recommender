from word_forms.word_forms import get_word_forms
from word_forms.lemmatizer import lemmatize

def generate_synonyms_for_emotion(emotion, synonyms_set):
    expanded_synonyms = set()
    for synonym in synonyms_set:
        expanded_synonyms.add(synonym)
        word_forms_dict = get_word_forms(synonym)
        all_related_forms = set()
        for pos_forms in word_forms_dict.values():
            all_related_forms.update(pos_forms)
        expanded_synonyms.update(all_related_forms)
    return expanded_synonyms

emotion_synonyms = {
    'sad': {'melancholy', 'sorrowful', 'mournful', 'depressed', 'dejected', 'blue', 'dismal', 'woeful', 'disheartened', 'crestfallen', 'unhappy', 'cry', 'gloomy', 'low', 'forlorn', 'downcast', 'grief', 'weeping', 'pessimistic', 'somber', 'bitter', 'tragic', 'upset', 'sob'},
    'love': {'affection', 'adoration', 'devotion', 'fondness', 'passion', 'romance', 'tenderness', 'intimacy', 'infatuation', 'amour', 'lust', 'sex', 'girlfriend', 'boyfriend', 'cherish', 'kiss', 'heart', 'relationship', 'fling', 'soulmate'},
    'anger': {'fury', 'wrath', 'rage', 'indignation', 'resentment', 'hostility', 'ire', 'annoyance', 'frustration', 'outrage', 'hatred', 'antagonism',' outrage', 'violence', 'temper', 'infuriation', 'displeasure'},
    'fear': {'dread', 'apprehension', 'anxiety', 'unease', 'terror', 'panic', 'phobia', 'fright', 'nervousness', 'alarm', 'suspicion', 'worry', 'angst', 'nightmare', 'creep', 'revulsion', 'fright', 'foreboding', 'disquieting'},
    'joy': {'happiness', 'delight', 'elation', 'bliss', 'jubilation', 'ecstasy', 'contentment', 'exhilaration', 'mirth', 'glee', 'laugh', 'liveliness', 'cheer', 'satisfaction', 'wonder', 'festivity', 'rejoicing', 'alleviation'}
}

expanded_emotion_synonyms = {}
for emotion, synonyms_set in emotion_synonyms.items():
    expanded_synonyms = generate_synonyms_for_emotion(emotion, synonyms_set)
    expanded_emotion_synonyms[emotion] = expanded_synonyms


