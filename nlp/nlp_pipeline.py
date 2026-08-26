#================================================================ Model Def:
'''
Natural Language Processing (NLP) is a field of artificial intelligence focused on the interaction between computers and human language. It involves tasks such as text processing, text analysis, and language understanding. Here's a breakdown of common NLP tasks and techniques:

    >   Text Preprocessing: Before performing NLP tasks, text data often requires preprocessing, which includes tokenization (splitting text into words or sentences), lowercasing, removing punctuation, stop word removal, and stemming or lemmatization to normalize words.

    >   Bag-of-Words (BoW) Representation: BoW is a simple and commonly used representation of text data, where each document is represented as a vector of word counts or term frequencies. This representation disregards word order and semantic meaning but is useful for tasks like sentiment analysis and document classification.

    >   TF-IDF (Term Frequency-Inverse Document Frequency): TF-IDF is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents. It combines term frequency (how often a word appears in a document) with inverse document frequency (how common or rare a word is across documents) to assign weights to words.

    >   Word Embeddings: Word embeddings are dense, low-dimensional vector representations of words learned from large text corpora using techniques like Word2Vec, GloVe, or FastText. Word embeddings capture semantic relationships between words and are widely used in NLP tasks like text classification, named entity recognition, and machine translation.

    >   Deep Learning for NLP: Deep learning models such as recurrent neural networks (RNNs), convolutional neural networks (CNNs), and transformer-based architectures like BERT and GPT have achieved state-of-the-art performance in various NLP tasks, including text generation, language translation, sentiment analysis, and question answering.

When to use NLP:

* Text Classification: NLP is commonly used for text classification tasks such as sentiment analysis, spam detection, topic classification, and language identification.
* Named Entity Recognition (NER): NLP can extract entities such as names, locations, organizations, and dates from text data, which is useful for information retrieval and entity linking.
* Machine Translation: NLP techniques are applied to machine translation systems that translate text from one language to another, facilitating cross-lingual communication.
* Information Extraction: NLP can extract structured information from unstructured text data, such as extracting keyphrases, relations between entities, and event extraction from news articles or social media posts.
'''

#================================================================ Template:
#====== Importing needed libraries:
import random
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

#====== Example NLP Pipeline:
# 1. Load the data
# 2. Preprocess the text (lowercase, strip punctuation, drop stop words)
# 3. Convert text into numerical features (TF-IDF)
# 4. Split into training and testing sets
# 5. Train a classifier
# 6. Evaluate

#====== Generate some example data (replace this with your actual data):
# For a real corpus, load it instead, e.g.:
#   reviews_df = pd.read_csv("reviews.csv")   # columns: review, sentiment
random.seed(42)

OPENERS = ["The film", "This movie", "The story", "The script", "The cast",
           "The direction", "The soundtrack", "The pacing"]
POSITIVE = ["was brilliant", "was superb", "was genuinely moving", "was excellent",
            "kept me hooked", "was beautifully crafted", "exceeded expectations"]
NEGATIVE = ["was dull", "was a mess", "dragged badly", "was disappointing",
            "felt lifeless", "wasted its premise", "fell completely flat"]
CLOSERS = ["and I would watch it again.", "from start to finish.", "throughout.",
           "in almost every scene.", "for the entire runtime.", ""]

rows = []
for _ in range(300):
    positive = random.random() < 0.5
    phrase = random.choice(POSITIVE if positive else NEGATIVE)
    text = f"{random.choice(OPENERS)} {phrase} {random.choice(CLOSERS)}".strip()
    rows.append({"review": text, "sentiment": "positive" if positive else "negative"})

reviews_df = pd.DataFrame(rows)

#====== Text preprocessing:
# NLTK ships a curated stop-word list. It needs a one-off corpus download, so we
# fall back to a minimal inline list when the data (or the network) is unavailable.
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = {"the", "a", "an", "and", "or", "but", "is", "was", "were", "be",
                  "been", "it", "its", "this", "that", "i", "would", "from", "in",
                  "for", "of", "to", "with", "on", "at", "as", "almost"}


def preprocess_text(text):
    """Lowercase, strip punctuation, and drop stop words."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(token for token in tokens if token not in STOP_WORDS)


reviews_df["cleaned_review"] = reviews_df["review"].apply(preprocess_text)

#====== Convert text into numerical features using TF-IDF:
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(reviews_df["cleaned_review"])
y = reviews_df["sentiment"]

#====== Split the dataset into training and testing sets:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#====== Train a Multinomial Naive Bayes classifier:
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

#====== Predict sentiment on the testing data:
y_pred = nb_classifier.predict(X_test)

#====== Evaluate the model:
# NOTE: the generated corpus above is trivially separable, so this scores ~100%.
# Real text data will not. Swap in a genuine dataset to see meaningful numbers.
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


#================================================================ Notes on Model construction:
'''
NLTK (Natural Language Toolkit):
NLTK is a popular Python library for NLP tasks, providing modules for text tokenization, stemming, lemmatization, part-of-speech tagging, and more.

TF-IDF Vectorizer:
TF-IDF vectorizer is a feature extraction technique used to convert text data into numerical features based on term frequency-inverse document frequency.

Multinomial Naive Bayes Classifier:
Multinomial Naive Bayes is a probabilistic classifier commonly used for text classification tasks, such as sentiment analysis and document classification.

Accuracy Score:
Accuracy score is a metric used to evaluate the performance of classification models, measuring the proportion of correctly classified instances out of all instances.
'''
