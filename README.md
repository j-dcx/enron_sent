# enron_sent
EMAIL SORTING SYSTEM

Purpose of this system is to categorize text with multi-classification machine learning algorithms / natural language processing in Python language.

DATA
- Dataset = https://www.cs.cmu.edu/~enron/
- Features = enron_mail_20150507/maildir/*/*sent* [all files found]
- Labels = enron_mail_20150507/maildir/[username]

DATA PREPROCESS
- Filtered subset containing 39624 emails (number of emails belonging to a specific user is in range 3000...9000)
- Translation from 'en' to 'fi', 'sv', 'no', 'de' using googletrans==3.1.0a0
- Stopwords extracted using Spacy lemmas, removed if found in nltk.stopwords of the specific language

VECTORIZATION
Formats
-- Word2Vec
--- Skip-gram
--- CBOW
-- Spacy
-- GloVe (not yet implemented)
--- glove.6B.300d
--- trained on data
-- sklearn HashingVectorizer  + TfidfTransformer
-- tensorflow.keras Tokenizer (technically not a vector, however was only format working somewhat ok with LSTM)
- Parameters
-- Dimension count = 300
-- Vocabulary size = 30000

MODEL TRAINING ATTRIBUTES
- Language
-- en
-- fi
-- sv
-- no
-- de
- Model type
-- Sequential [three types of architectures]
--- MLP (Multi Layer Perceptron)
--- CNN (Convolutional Neuron Network)
--- LSTM (Long-Short Term Memory)
-- Non-Sequential
--- [13 variations of scikit-learn machine learning algorithms - not yet implemented]

FINE-TUNING (sequential models)
- Early stopping
-- monitor = 'val_loss'
-- patience = 5
- Optimizer (either of the following)
-- SGD
-- Adam

METRICS
- loss
- accuracy
- precision
- recall
- f1
- jaccard score
- auc
- categorical accuracy
- categorical crossentropy
- categorical hinge

RESULTS
- Best results achieved by
-- Language = en
-- Vector = Word2Vec+skip-gram
-- Model = MLP
-- Optimizer = SGD
