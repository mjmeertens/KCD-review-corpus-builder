# Import stuff
import json
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from tqdm import tqdm
import os 
from spacy.cli import download as spacy_download
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model (en_core_web_sm)...")
    spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# making a path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
def dpath(file):
    return os.path.join(DATA_DIR, file)

# Setup
def ensure_nltk_resource(resource_path, download_name):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        print(f"Downloading NLTK resource: {download_name}...")
        nltk.download(download_name)

ensure_nltk_resource('tokenizers/punkt', 'punkt')
ensure_nltk_resource('corpora/stopwords', 'stopwords')

stop_words = set(stopwords.words('english'))
stop_words.discard('not')
tqdm.pandas()

# Function 1: clean reviews
def clean_reviews(df):
    df = df[["review"]]
    df = df.dropna(subset=["review"])
    df = df[df["review"].str.strip() != ""]
    return df

# Function 2: full preprocessing pipeline
def preprocess(text):
    original_text = str(text)
    
    # replace newlines with spaces, tokenizing
    text = original_text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = word_tokenize(text)

    # into lemma
    doc = nlp(" ".join(tokens))
    lemmas = [token.lemma_.lower() for token in doc]

    # remove stopwords (except "not")
    tokens_no_stop = [word for word in lemmas if word not in stop_words]

    # sentence splitting
    sentences = sent_tokenize(original_text)

    return pd.Series([lemmas, tokens_no_stop, sentences])

# files processing loop
files = [
    ("review_379430.json", "kcd1_processed.csv"),
    ("review_1771300.json", "kcd2_processed.csv")
]

for input_file, output_file in files:
    
    with open(dpath(input_file), "r") as f:
        data = json.load(f)

    df = pd.DataFrame.from_dict(data["reviews"], orient="index")

    df = clean_reviews(df)

    # Apply full pipeline
    df[["tokens_with_stopwords", "tokens_no_stopwords", "sentences"]] = df["review"].progress_apply(preprocess)

    # Save
    df.to_csv(dpath(output_file), index=False)

    print(f"Saved {len(df)} reviews to {dpath(output_file)}")