import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import string
punctuations = string.punctuation

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
import spacy
import streamlit as st
import tensorflow as tf

from collections import defaultdict
from gensim.models.word2vec import Word2Vec
from keras_preprocessing.sequence import pad_sequences
from matplotlib.pyplot import figure
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

global GLOBAL_PATH
#GLOBAL_PATH = '/content/drive/MyDrive/Colab Notebooks/final/'
GLOBAL_PATH = ''
global DF_LEN
DF_LEN = {'en':39624,'fi':39624,'sv':39624,'no':39624,'de':39624}

#*************************** BEGIN ****************************************
st.write("""
## E-mail Sorting System
###### Parameters
""")

st.sidebar.header('Parameter input')


#*************************** GET LANGUAGE ****************************************
def user_input_features1():
    model_languages = {'English':'en','Finnish':'fi','Swedish':'sv','Norwegian':'no','German':'de'}

    model_language = st.sidebar.selectbox('Model Language', ('English','Finnish','Swedish','Norwegian','German'))

    data = {'model_language': model_languages[model_language]}
    return pd.DataFrame(data, index=[0])
#*************************** user_input_features_df1 ****************************************
user_input_features_df1 = user_input_features1()
model_language = user_input_features_df1['model_language'].values[0]

st.write(user_input_features_df1)



#*************************** IMPORT GLOBALS AND FUNCTIONS ****************************************
import my_functions as f


#*************************** Initialize variables ****************************************
text = ''
msg = ''
files = os.listdir(GLOBAL_PATH + model_language + '/' + f.GRAPH_DATA_DIRECTORY)
graph_data_found = False
spacy_packages = {'en':'en_core_web_sm-3.4.0',
                        'fi':'fi_core_news_lg-3.4.0',
                        'sv':'sv_core_news_lg-3.4.0',
                        'no':'nb_core_news_lg-3.4.0',
                        'de':'de_core_news_lg-3.4.0'}


#*************************** user_input_features_df2 ****************************************
user_input_features_df2 = f.user_input_features2(spacy_packages[model_language])
model_type = user_input_features_df2['model_type'].values[0]
vector_type = user_input_features_df2['vector_type'].values[0]
model_architecture = user_input_features_df2['model_architecture'].values[0]
optimizer = user_input_features_df2['optimizer'].values[0]
graph_data = model_architecture + '_' + vector_type + '_' + str(30) + 'k_' + str(300) + 'd_' + optimizer + '_' + model_language

st.write(user_input_features_df2)


#*************************** user_input_features_df3 ****************************************
graph_data_found = False
if graph_data in files:
    graph_data_found = True


user_input_features_df3 = f.user_input_features3(GLOBAL_PATH, model_language, graph_data, graph_data_found, DF_LEN)
metric = user_input_features_df3['metric'].values[0]
vector_dimension = user_input_features_df3['vector_dimension'].values[0]
dataset_size = user_input_features_df3['dataset_size'].values[0]
vocabulary_size = (int)(user_input_features_df3['vocabulary_size'].values[0] / 1000)
graph_data = model_architecture + '_' + vector_type + '_' + str(vocabulary_size) + 'k_' + str(vector_dimension) + 'd_' + optimizer + '_' + model_language

st.write(user_input_features_df3)


if graph_data_found:
    st.write('Available graph data with lang = ', model_language, ':', files)
    st.write('Using graph data [' + graph_data + ']')

    #*************************** Load CSV and Spacy ****************************************
    df = f.load_dataset(GLOBAL_PATH, model_language)
    nlp = spacy.load(GLOBAL_PATH + model_language + '/spacy/' + spacy_packages[model_language])


    #*************************** Plot graph ****************************************
    f.plot_scores(GLOBAL_PATH, model_language, graph_data, [user_input_features_df1, user_input_features_df2, user_input_features_df3], metric)
    f.print_results(pickle.load(open(GLOBAL_PATH + model_language + '/results/' + graph_data, 'rb')))

    #*************************** Print user value counts ****************************************
    st.write('User value counts:',df.user.value_counts())
    users = df.user.unique().tolist()

    with st.form("my_form_text"):
        seed_max = dataset_size - 1 if dataset_size else len(df) - 1


        #*************************** Input email text with random seed ****************************************
        submit_random_seed = st.form_submit_button("Generate random seed")
        if submit_random_seed:
            random_seed = np.random.randint(0, seed_max)
            text_header = "Random email seed [" + str(random_seed) + "] content:"
            text = df.iloc[random_seed].translation


        #*************************** Input email text with seed number ****************************************
        custom_seed = st.text_input("Seed number (choose between 1 and " + str(seed_max) + ")")
        submit_custom_seed = st.form_submit_button("Submit custom seed")
        if submit_custom_seed:
            if int(custom_seed) > seed_max:
                msg = "Seed number exceeded dataset maximum"
            else:
                text_header = "Custom email seed [" + str(custom_seed) + "] content in '" + model_language + "':"
                text = df.iloc[int(custom_seed)].translation


        #*************************** Input custom text ****************************************
        custom_text = st.text_input("Custom email content")
        submit_custom_text = st.form_submit_button("Submit text")
        if submit_custom_text:
            text_header = "Custom email content in '" + model_language + "':"
            text = custom_text


        #*************************** Input generated text ****************************************
        generate_length = st.text_input("or choose length of generated text")
        submit_generate = st.form_submit_button("Submit length")
        if submit_generate:
        
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
            from transformers import Trainer, TrainingArguments, GPT2LMHeadModel
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            model_generate = GPT2LMHeadModel.from_pretrained(GLOBAL_PATH + model_language + "/" + f.GENERATE_DIRECTORY)
            from transformers import pipeline

            generate = pipeline('text-generation', model=model_generate, tokenizer='gpt2', config={'max_length':10000, 'temperature': .5})
            initial_seed = df.sample(1).translation_clean.values[0][:100]
            st.write('initial seed: [', initial_seed, ']')
        
            st.write('generate length: ',generate_length)
            text = generate(initial_seed, max_length=int(generate_length)+10, min_length=int(generate_length), do_sample=True, temperature=1.2)[0]['generated_text']
            text = ' '.join(text.split()[:int(generate_length)-1])
            text_header = "Generated content in '" + model_language + "':"
            

    
    if msg != '':
        st.write('text: [', text, ']')

    if text != '':
        st.write(text_header)
        text_clean = f.cleanup_text(GLOBAL_PATH, model_language, spacy_packages, text)
        st.write('text_clean: [', text_clean, ']')
    
        # ************************************ Select model ************************************
        models_ann = model_architecture + '_' + vector_type + '_' + str(vocabulary_size) + 'k_' + str(vector_dimension) + 'd_' + optimizer + '_' + model_language
        model = keras.models.load_model(model_language + '/' + f.MODELS_ANN_DIRECTORY + models_ann + '/',custom_objects={'get_f1':f.get_f1,'jaccard_score':f.jaccard_score})
        model_input = []

        st.write('vector_type = ' + vector_type)
        
        if vector_type == 'tfkerastokenizer':
        
            tokenizer = Tokenizer(num_words=30000, lower=True)
            tokenizer.fit_on_texts(text)
            model_input = tokenizer.texts_to_sequences(text)
            model_input = [x for sublist in model_input for x in sublist]
            model_input = pad_sequences([model_input], maxlen=300, padding='post')

        elif vector_type == 'word2vec_skipgram' or vector_type == 'word2vec_cbow':
            with open(model_language + '/' + f.MODELS_VECTORS_DIRECTORY + vector_type + '_' + str(vocabulary_size) + 'k_' + str(vector_dimension) + 'd_' + model_language, 'rb') as f:
                vector_model = pickle.load(f)
            text_dim = 300
            model_input = np.zeros((1,text_dim), dtype='float32')
            num_words = 0.
            for word in text_clean.split():
                if word in vector_model.wv.key_to_index:
                    model_input = np.add(model_input, vector_model.wv.get_vector(word))
                    num_words += 1.
            if num_words != 0.:
                model_input = np.divide(model_input, num_words)

        elif vector_type == 'spacy':
            nlp = spacy.load(spacy_packages[model_language])
            for doc in nlp.pipe(text_clean):
                if doc.has_vector:
                    model_input.append(doc.vector)
                # If doc doesn't have a vector, then fill it with zeros.
                else:
                    model_input.append(np.zeros((128,), dtype="float32"))
                    
            model_input = np.array(model_input)
        
        elif vector_type == 'tfidf_hashing_vectorizer':
            from sklearn.feature_extraction.text import HashingVectorizer
            features_hash = HashingVectorizer(ngram_range=(1,2),n_features=300).transform([text])
            features_matrix = features_hash
            
            from sklearn.feature_extraction.text import TfidfTransformer
            tfidf_transformer = TfidfTransformer().fit(features_matrix)

            model_input = tfidf_transformer.transform(features_matrix).toarray()

        else:
            st.write('Vector type not set')
                    
        # ************************************ Predict ************************************
        st.write('model_input = [', model_input, ']')
        pred = model.predict(model_input)
        st.write('predictions (%): ',model.predict(model_input)*100)
        user_number = (np.argmax(model.predict(model_input)[0]))
        st.write('MAX[',user_number,'] = ',max(model.predict(model_input)[0])*100,'%')
        st.write('Predicted user:')
        st.write('users[',user_number,'] = ',users[user_number],'')


        # ************************************ Show and compare predictions ************************************
        if submit_random_seed:
            st.write('Actual user:')
            st.write(f'df[',random_seed,'] = ',df.iloc[random_seed].user)

            
        if submit_custom_seed:
            st.write('Actual user:')
            st.write(f'df[',custom_seed,'] = ',df.iloc[int(custom_seed)].user)
            
else:
    st.write('Graph data [' + graph_data + '] not found')