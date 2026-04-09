#By M.J. Meertens 2026

# imports
import pandas as pd
import ast
from collections import Counter
import spacy
from spacy.cli import download as spacy_download
try: nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
from tqdm import tqdm
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

# setting up
tqdm.pandas()
sia = SentimentIntensityAnalyzer()
def ensure_nltk_resource(resource_path, download_name):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        print(f"Downloading NLTK resource: {download_name}...")
        nltk.download(download_name)

ensure_nltk_resource('sentiment/vader_lexicon', 'vader_lexicon')
ensure_nltk_resource('corpora/sentiwordnet', 'sentiwordnet')
ensure_nltk_resource('corpora/wordnet', 'wordnet')
lemmatizer = WordNetLemmatizer()

# making a path for data, cache, and output. we're creating quite some files so i want it to keep organized.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
OUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def cpath(file): return os.path.join(CACHE_DIR, file)
def dpath(file): return os.path.join(DATA_DIR, file)
def opath(file): return os.path.join(OUT_DIR, file)

kcd1 = pd.read_csv(dpath("kcd1_processed.csv"))
kcd2 = pd.read_csv(dpath("kcd2_processed.csv"))

# POS tagging
if os.path.exists(cpath("kcd1_pos.pkl")) and os.path.exists(cpath("kcd2_pos.pkl")):
    print("Loading POS-tagged data...")
    kcd1 = pd.read_pickle(cpath("kcd1_pos.pkl"))
    kcd2 = pd.read_pickle(cpath("kcd2_pos.pkl"))

else:
    print("Running POS-tagging...")

    kcd1['tokens_with_stopwords'] = kcd1['tokens_with_stopwords'].apply(ast.literal_eval)
    kcd2['tokens_with_stopwords'] = kcd2['tokens_with_stopwords'].apply(ast.literal_eval)

    def pos_tag_tokens(token_list):
        doc = nlp(" ".join(token_list))
        return [(token.text, token.pos_) for token in doc]

    kcd1['pos_tags'] = kcd1['tokens_with_stopwords'].progress_apply(pos_tag_tokens)
    kcd2['pos_tags'] = kcd2['tokens_with_stopwords'].progress_apply(pos_tag_tokens)

    # ONLY save to cache
    kcd1.to_pickle(cpath("kcd1_pos.pkl"))
    kcd2.to_pickle(cpath("kcd2_pos.pkl"))


# reading tokens (with stopwords) for frequency and collocate analysis. we save this as a separate variable to avoid having to eval() the string multiple times in later steps.
if os.path.exists(cpath("kcd1_tokens.pkl")) and os.path.exists(cpath("kcd2_tokens.pkl")):
    print("Loading tokenized data...")
    kcd1_tokens = pd.read_pickle(cpath("kcd1_tokens.pkl"))
    kcd2_tokens = pd.read_pickle(cpath("kcd2_tokens.pkl"))

else:
    print("Converting tokens (first run)...")

    def safe_eval(x):
        return ast.literal_eval(x) if isinstance(x, str) else x

    kcd1['tokens_no_stopwords'] = kcd1['tokens_no_stopwords'].apply(safe_eval)
    kcd2['tokens_no_stopwords'] = kcd2['tokens_no_stopwords'].apply(safe_eval)

    kcd1_tokens = kcd1['tokens_no_stopwords']
    kcd2_tokens = kcd2['tokens_no_stopwords']

    pd.to_pickle(kcd1_tokens, cpath("kcd1_tokens.pkl"))
    pd.to_pickle(kcd2_tokens, cpath("kcd2_tokens.pkl"))


# frequency analysis
def normalized_freq(tokens_list, per=1000):
    all_tokens = [token for sublist in tokens_list for token in sublist]
    total_tokens = len(all_tokens)
    counts = Counter(all_tokens)

    return dict(sorted({
        word: (count / total_tokens) * per
        for word, count in counts.items()
    }.items(), key=lambda x: x[1], reverse=True))

if os.path.exists(cpath("freq_kcd1.pkl")) and os.path.exists(cpath("freq_kcd2.pkl")):
    print("Loading frequency results...")
    freq_kcd1_1k = pd.read_pickle(cpath("freq_kcd1.pkl"))
    freq_kcd2_1k = pd.read_pickle(cpath("freq_kcd2.pkl"))

else:
    print("Computing frequency analysis...")
    freq_kcd1_1k = normalized_freq(kcd1_tokens)
    freq_kcd2_1k = normalized_freq(kcd2_tokens)

    pd.to_pickle(freq_kcd1_1k, cpath("freq_kcd1.pkl"))
    pd.to_pickle(freq_kcd2_1k, cpath("freq_kcd2.pkl"))


def print_top_words(freq_dict, title, n=20):
    print(f"\n{title}")
    print("-" * 40)
    print(f"{'Rank':<5}{'Word':<15}{'Freq (per 1k)':>15}")
    print("-" * 40)
    
    for i, (word, freq) in enumerate(list(freq_dict.items())[:n], start=1):
        print(f"{i:<5}{word:<15}{freq:>15.2f}")

print_top_words(freq_kcd1_1k, "KCD1 Top Words")
print_top_words(freq_kcd2_1k, "KCD2 Top Words")

#evaluative language extraction
def get_wordnet_pos(tag):
    if tag == "ADJ":
        return wn.ADJ
    elif tag == "ADV":
        return wn.ADV
    elif tag == "VERB":
        return wn.VERB
    return None


def lemmatize_word(word, tag):
    wn_tag = get_wordnet_pos(tag)
    if wn_tag:
        return lemmatizer.lemmatize(word, pos=wn_tag)
    return word


def get_swn_score(word, tag):
    wn_tag = get_wordnet_pos(tag)
    if wn_tag is None:
        return 0

    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return 0

    scores = []
    for syn in synsets:
        try:
            swn_syn = swn.senti_synset(syn.name())
            score = swn_syn.pos_score() - swn_syn.neg_score()
            scores.append(score)
        except:
            continue

    if scores:
        return sum(scores) / len(scores)
    return 0



def extract_pos(pos_series, tag, exclude=None, use_swn=False, threshold=0.05):
    words = []

    for doc in pos_series:
        for word, t in doc:
            if t == tag:

                # lemmatize
                word = lemmatize_word(word, t)

                # remove non-evaluative verbs
                if exclude and word in exclude:
                    continue

                # SentiWordNet filter
                if use_swn:
                    score = get_swn_score(word, t)

                    if abs(score) < threshold:
                        continue

                words.append(word)

    return Counter(words)





adj_kcd1 = extract_pos(kcd1['pos_tags'], "ADJ", use_swn=True)
adv_kcd1 = extract_pos(kcd1['pos_tags'], "ADV", use_swn=True)
verb_kcd1 = extract_pos(kcd1['pos_tags'], "VERB", use_swn=True)


adj_kcd2 = extract_pos(kcd2['pos_tags'], "ADJ", use_swn=True)
adv_kcd2 = extract_pos(kcd2['pos_tags'], "ADV", use_swn=True)
verb_kcd2 = extract_pos(kcd2['pos_tags'], "VERB", use_swn=True)


def build_eval_table(adj_counter, adv_counter, verb_counter,
                     total_tokens, title, n=20):

    adj = adj_counter.most_common(n)
    adv = adv_counter.most_common(n)
    verb = verb_counter.most_common(n)

    print(f"\n{title}")
    print("-" * 80)
    print(f"{'Rank':<5}"
          f"{'Adjective':<15}{'Per 1k':<10}"
          f"{'Adverb':<15}{'Per 1k':<10}"
          f"{'Verb':<15}{'Per 1k':<10}")
    print("-" * 80)

    for i in range(n):
        adj_word, adj_count = adj[i] if i < len(adj) else ("", 0)
        adv_word, adv_count = adv[i] if i < len(adv) else ("", 0)
        verb_word, verb_count = verb[i] if i < len(verb) else ("", 0)

        adj_freq = (adj_count / total_tokens) * 1000
        adv_freq = (adv_count / total_tokens) * 1000
        verb_freq = (verb_count / total_tokens) * 1000

        print(f"{i+1:<5}"
              f"{adj_word:<15}{adj_freq:<10.2f}"
              f"{adv_word:<15}{adv_freq:<10.2f}"
              f"{verb_word:<15}{verb_freq:<10.2f}")




total_kcd1 = sum(len(x) for x in kcd1_tokens)
total_kcd2 = sum(len(x) for x in kcd2_tokens)


build_eval_table(adj_kcd1, adv_kcd1, verb_kcd1, total_kcd1, "KCD1 Evaluative Language")

build_eval_table(adj_kcd2, adv_kcd2, verb_kcd2,total_kcd2, "KCD2 Evaluative Language")

# looking at collocates
if os.path.exists(cpath("collocates.pkl")):
    print("Loading collocates (pickle)...")
    coll_df = pd.read_pickle(cpath("collocates.pkl"))
    print("Loaded collocates instantly.")

else:
    print("Computing collocates (this may take a while)...")

    aspects = {
        "gameplay": ["mission", "item", "map", "weapon", "mode", "multiplayer", "quest", "combat", "story", "gameplay"],
        "graphics": ["graphic", "graphics", "visual", "look", "aesthetic", "animation", "frame", "design", "art", "world"],
        "audio": ["audio", "sound", "music", "soundtrack", "melody", "voice", "valta"],
        "community": ["community", "support", "toxic", "friendly", "player", "players", "developer", "devs"],
        "performance": ["server", "bug", "bugs", "connection", "lag", "latency", "ping", "crash", "crashes", "glitch", "glitches", "fps", "framerate"]
    }



    def get_word_collocates(tokens_series, pos_series, target, window=2):
        collocates = []

        for tokens, pos_tags in zip(tokens_series, pos_series):
            min_len = min(len(tokens), len(pos_tags))

            for i in range(min_len):
                if tokens[i] == target:
                    start = max(i - window, 0)
                    end = min(i + window + 1, min_len)

                    for j in range(start, end):
                        if j != i:
                            w, tag = pos_tags[j]
                            if tag in ["ADJ", "ADV", "NOUN", "VERB"]:
                                collocates.append((w, tag))

        return Counter(collocates)

    rows = []

    for corpus_name, tokens_series, pos_series in [
        ("KCD1", kcd1_tokens, kcd1['pos_tags']),
        ("KCD2", kcd2_tokens, kcd2['pos_tags'])
    ]:
        print(f"Processing {corpus_name}...")

        for aspect, words in aspects.items():
            for word in words:

                coll = get_word_collocates(tokens_series, pos_series, word)

                # POS
                pos_groups = {
                    "ADJ": [],
                    "ADV": [],
                    "NOUN": [],
                    "VERB": []
                }

                for (w, tag), count in coll.items():
                    if tag in pos_groups:
                        pos_groups[tag].append((w, count))

                # Top 5 
                for pos, items in pos_groups.items():
                    top5 = sorted(items, key=lambda x: x[1], reverse=True)[:5]

                    for collocate, count in top5:
                        rows.append({
                            "corpus": corpus_name,
                            "aspect": aspect,
                            "word": word,
                            "pos": pos,
                            "collocate": collocate,
                            "count": count
                        })

    coll_df = pd.DataFrame(rows)

    coll_df = coll_df.sort_values(
        by=["corpus", "aspect", "word", "pos", "count"],
        ascending=[True, True, True, True, False]
    )

    print("Saving collocates (Excel + pickle)...")

    coll_df.to_pickle(cpath("collocates.pkl"))

    with pd.ExcelWriter(opath("collocates.xlsx")) as writer:
        coll_df[coll_df["corpus"] == "KCD1"].to_excel(writer, sheet_name="KCD1", index=False)
        coll_df[coll_df["corpus"] == "KCD2"].to_excel(writer, sheet_name="KCD2", index=False)

    print("Saved. Future runs will be instant.")

# sentiment on aspect
aspects = {
        "gameplay": ["mission", "item", "map", "weapon", "mode", "multiplayer", "quest", "combat", "story", "gameplay"],
        "graphics": ["graphic", "graphics", "visual", "look", "aesthetic", "animation", "frame", "design", "art", "world"],
        "audio": ["audio", "sound", "music", "soundtrack", "melody", "voice", "valta"],
        "community": ["community", "support", "toxic", "friendly", "player", "players", "developer", "devs"],
        "performance": ["server", "bug", "bugs", "connection", "lag", "latency", "ping", "crash", "crashes", "glitch", "glitches", "fps", "framerate"]
    }


if os.path.exists(cpath("aspect_sentiment.pkl")):
    print("Loading aspect sentiment (pickle)...")
    sent_df = pd.read_pickle(cpath("aspect_sentiment.pkl"))
    print("Loaded instantly.")


else:
    print("Computing aspect sentiment...")

    def get_all_aspect_collocates(tokens_series, pos_series, targets, window=2):
        collocates = []

        for tokens, pos_tags in zip(tokens_series, pos_series):
            min_len = min(len(tokens), len(pos_tags))

            for i in range(min_len):
                if tokens[i] in targets:
                    start = max(i - window, 0)
                    end = min(i + window + 1, min_len)

                    for j in range(start, end):
                        if j != i:
                            w, tag = pos_tags[j]
                            if tag in ["ADJ", "ADV", "NOUN", "VERB"]:
                                collocates.append(w)

        return Counter(collocates)

    def aspect_sentiment_stats(counter):
        scores = []

        for word, count in counter.items():
            score = sia.polarity_scores(word)['compound']
            scores.extend([score] * count)

        if len(scores) == 0:
            return 0, 0

        return np.mean(scores), np.std(scores)

    rows = []

    for aspect, words in aspects.items():

        coll1 = get_all_aspect_collocates(kcd1_tokens, kcd1['pos_tags'], words)
        coll2 = get_all_aspect_collocates(kcd2_tokens, kcd2['pos_tags'], words)

        mean1, std1 = aspect_sentiment_stats(coll1)
        mean2, std2 = aspect_sentiment_stats(coll2)

        rows.append({
            "aspect": aspect,
            "KCD1_mean": mean1,
            "KCD1_std": std1,
            "KCD2_mean": mean2,
            "KCD2_std": std2,
            "difference": mean2 - mean1
        })

    sent_df = pd.DataFrame(rows)

    print("Saving sentiment results...")

    sent_df.to_pickle(cpath("aspect_sentiment.pkl"))                  
    sent_df.to_excel(opath("aspect_sentiment.xlsx"), index=False)    

    print("Saved. Future runs will be instant.")

## VADER sentiment analysis on full review
if os.path.exists(cpath("review_sentiment.pkl")):
    print("Loading review-level sentiment (pickle)...")
    review_df = pd.read_pickle(cpath("review_sentiment.pkl"))
    print("Loaded instantly.")

else:
    print("Computing review-level sentiment...")

    def review_sentiment(text):
        score = sia.polarity_scores(str(text))['compound']
        if score > 0:
            label = "positive"
        elif score < 0:
            label = "negative"
        else:
            label = "neutral" 
        return pd.Series([score, label])

    kcd1[['review_score', 'review_label']] = kcd1['review'].progress_apply(review_sentiment)
    kcd2[['review_score', 'review_label']] = kcd2['review'].progress_apply(review_sentiment)


    kcd1_filtered = kcd1[kcd1['review_label'] != "neutral"]
    kcd2_filtered = kcd2[kcd2['review_label'] != "neutral"]

    review_df = pd.DataFrame({
        "corpus": ["KCD1", "KCD2"],
        "mean_sentiment": [kcd1['review_score'].mean(), kcd2['review_score'].mean()],
        "std_sentiment": [kcd1['review_score'].std(), kcd2['review_score'].std()],
        "positive_%": [
            (kcd1_filtered['review_label'] == "positive").mean(),
            (kcd2_filtered['review_label'] == "positive").mean()
        ],
        "negative_%": [
            (kcd1_filtered['review_label'] == "negative").mean(),
            (kcd2_filtered['review_label'] == "negative").mean()
        ]
    })

    print("Saving review-level sentiment...")
    review_df.to_pickle(cpath("review_sentiment.pkl"))
    review_df.to_excel(opath("review_sentiment.xlsx"), index=False)

    print("Saved review-level sentiment.")


# sentence level. 

if os.path.exists(cpath("sentence_sentiment.pkl")):
    print("Loading sentence-level sentiment (pickle)...")
    sentence_df = pd.read_pickle(cpath("sentence_sentiment.pkl"))
    print("Loaded instantly.")

else:
    print("Computing sentence-level sentiment...")

    kcd1['sentences'] = kcd1['sentences'].apply(ast.literal_eval)
    kcd2['sentences'] = kcd2['sentences'].apply(ast.literal_eval)

    def sentence_scores(sent_list):
        scores = []
        for s in sent_list:
            score = sia.polarity_scores(str(s))['compound']
            scores.append(score)
        return scores

    kcd1['sentence_scores'] = kcd1['sentences'].progress_apply(sentence_scores)
    kcd2['sentence_scores'] = kcd2['sentences'].progress_apply(sentence_scores)


    kcd1_all = [score for sublist in kcd1['sentence_scores'] for score in sublist]
    kcd2_all = [score for sublist in kcd2['sentence_scores'] for score in sublist]


    kcd1_filtered = [s for s in kcd1_all if s != 0]
    kcd2_filtered = [s for s in kcd2_all if s != 0]

    def polarity_dist(scores):
        pos = sum(1 for s in scores if s > 0)
        neg = sum(1 for s in scores if s < 0)
        total = len(scores)
        return pos/total, neg/total

    pos1, neg1 = polarity_dist(kcd1_filtered)
    pos2, neg2 = polarity_dist(kcd2_filtered)

    sentence_df = pd.DataFrame({
        "corpus": ["KCD1", "KCD2"],
        "mean_sentiment": [np.mean(kcd1_all), np.mean(kcd2_all)], 
        "std_sentiment": [np.std(kcd1_all), np.std(kcd2_all)],
        "positive_%": [pos1, pos2],
        "negative_%": [neg1, neg2]
    })

    print("Saving sentence-level sentiment...")
    sentence_df.to_pickle(cpath("sentence_sentiment.pkl"))
    sentence_df.to_excel(opath("sentence_sentiment.xlsx"), index=False)

    print("Saved sentence-level sentiment.")
