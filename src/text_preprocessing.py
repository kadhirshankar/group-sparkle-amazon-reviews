import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split


#words sklearn's English stop list removes that matter for sentiment
KEEP_WORDS = {'not', 'no', 'nor', 'never', 'nothing', 'none', 'cannot', 'without',
              'but', 'however', 'very', 'too', 'few', 'less', 'more', 'most', 'only'}
STOP_WORDS = list(ENGLISH_STOP_WORDS - KEEP_WORDS)


#lowercase, remove HTML tags like <br />, and turn "isn't" into "is not"
def clean_text(text):
    text = text.lower().replace("’", "'")
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace("won't", "will not").replace("can't", "can not")
    text = text.replace("n't", " not")
    return text


#1-2 stars = Negative, 3 = Neutral, 4-5 = Positive
def score_to_sentiment(score):
    if score <= 2:
        return 'Negative'
    if score == 3:
        return 'Neutral'
    return 'Positive'


#load the sample, remove duplicate reviews, then split 80/20 (stratified, seed 49)
def load_and_split(path):
    df = pd.read_csv(path)
    required = {'Score', 'Text'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df.dropna(subset=['Text']).drop_duplicates(subset='Text').copy()
    df['sentiment'] = df['Score'].apply(score_to_sentiment)
    train, test = train_test_split(df, test_size=0.2, stratify=df['sentiment'], random_state=49)
    return train, test


#BOW or TF-IDF vectorizer: cleans text, keeps negations, adds two-word phrases ("not good")
def make_vectorizer(kind):
    settings = dict(preprocessor=clean_text, stop_words=STOP_WORDS,
                    ngram_range=(1, 2), max_features=10000)
    if kind == 'bow':
        return CountVectorizer(**settings)
    if kind == 'tfidf':
        return TfidfVectorizer(**settings)
    raise ValueError("kind must be 'bow' or 'tfidf'")
