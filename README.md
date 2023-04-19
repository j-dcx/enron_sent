### https://www.emailsorter.me
EMAIL SORTING SYSTEM

Purpose of this system is to categorize text with multi-classification machine learning algorithms / natural language processing in Python language.

- DATA
&nbsp;- Dataset = https://www.cs.cmu.edu/~enron/<br>
&nbsp;- Features = enron_mail_20150507/maildir/*/*sent* [all files found]<br>
&nbsp;- Labels = enron_mail_20150507/maildir/[username]<br>

- DATA PREPROCESS
&nbsp;- Filtered subset containing 39624 emails (number of emails belonging to a specific user is in range 3000...9000<br>
&nbsp;- Translation from 'en' to 'fi', 'sv', 'no', 'de' using googletrans==3.1.0a0<br>
&nbsp;- Stopwords extracted using Spacy lemmas, removed if found in nltk.stopwords of the specific language<br>

- VECTORIZATION
&nbsp;- Formats<br>
&nbsp;---- Word2Vec<br>
&nbsp;-------- Skip-gram<br>
&nbsp;-------- CBOW<br>
&nbsp;---- Spacy<br>
&nbsp;---- GloVe (not yet implemented)<br>
&nbsp;-------- glove.6B.300d (https://nlp.stanford.edu/projects/glove/)<br>
&nbsp;-------- trained on data<br>
&nbsp;---- sklearn HashingVectorizer  + TfidfTransformer<br>
&nbsp;---- tensorflow.keras Tokenizer (technically not a vector, however was only format working somewhat ok with LSTM)<br>
&nbsp;- Parameters<br>
&nbsp;---- Dimension count = 300<br>
&nbsp;---- Vocabulary size = 30000<br>

- MODEL TRAINING ATTRIBUTES
&nbsp;- Language<br>
&nbsp;---- en<br>
&nbsp;---- fi<br>
&nbsp;---- sv<br>
&nbsp;---- no<br>
&nbsp;---- de<br>
&nbsp;- Model type<br>
&nbsp;---- Sequential [three types of architectures]<br>
&nbsp;-------- MLP (Multi Layer Perceptron)<br>
&nbsp;-------- CNN (Convolutional Neuron Network)<br>
&nbsp;-------- LSTM (Long-Short Term Memory)<br>
&nbsp;---- Non-Sequential<br>
&nbsp;-------- [13 variations of scikit-learn machine learning algorithms - not yet implemented]<br>

- FINE-TUNING (sequential models)<br>
&nbsp;- Early stopping<br>
&nbsp;---- monitor = 'val_loss'<br>
&nbsp;---- patience = 5<br>
&nbsp;- Optimizer (either of the following)<br>
&nbsp;---- SGD<br>
&nbsp;---- Adam<br>

- METRICS<br>
&nbsp;- loss<br>
&nbsp;- accuracy<br>
&nbsp;- precision<br>
&nbsp;- recall<br>
&nbsp;- f1<br>
&nbsp;- jaccard score<br>
&nbsp;- auc<br>
&nbsp;- categorical accuracy<br>
&nbsp;- categorical crossentropy<br>
&nbsp;- categorical hinge<br>

- RESULTS<br>
&nbsp;- Best results achieved by<br>
&nbsp;---- Language = en<br>
&nbsp;---- Vector = Word2Vec+skip-gram<br>
&nbsp;---- Model = MLP<br>
&nbsp;---- Optimizer = SGD<br>
