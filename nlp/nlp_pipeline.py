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
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#====== Example NLP Pipeline:
# Load data
# Perform text preprocessing (tokenization, lowercasing, stop word removal, etc.)
# Convert text data into numerical features (e.g., using TF-IDF)
# Split the dataset into training and testing sets
# Train a machine learning model on the training data
# Evaluate the model on the testing data

# Importing needed libraries
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load IMDb movie reviews dataset
imdb_df = pd.read_csv('imdb_reviews.csv')

# Text Preprocessing
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())  # Tokenization and lowercasing
    tokens = [token for token in tokens if token.isalnum()]  # Remove punctuation
    tokens = [token for token in tokens if token not in stop_words]  # Remove stop words
    return ' '.join(tokens)

imdb_df['cleaned_review'] = imdb_df['review'].apply(preprocess_text)

# Convert text data into numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(imdb_df['cleaned_review'])
y = imdb_df['sentiment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predict sentiment on the testing data
y_pred = nb_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


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
