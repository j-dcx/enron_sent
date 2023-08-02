import matplotlib.pyplot as plt
import pandas as pd
import pickle
import spacy
import streamlit as st

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import string
punctuations = string.punctuation

global CSV
global MODELS_VECTOR_DIRECTORY
global MODELS_ANN_DIRECTORY
global MODELS_ANN_DEFAULT
global GRAPH_DATA_DIRECTORY
global GRAPH_DATA_DEFAULT
global SPACY_PACKAGE_DEFAULT
global GENERATE_DIRECTORY


MODELS_VECTORS_DIRECTORY = 'models/vector/'
MODELS_ANN_DIRECTORY = 'models/ann/'
GRAPH_DATA_DIRECTORY = 'graph_data/'
GENERATE_DIRECTORY = 'generate/'
CSV = 'data/df_'

MODELS_ANN_DEFAULT = 'mlp_word2vec_skipgram_30k_300d_sgd_en'
GRAPH_DATA_DEFAULT = 'mlp_word2vec_skipgram_30k_300d_sgd_en'
SPACY_PACKAGE_DEFAULT = 'en_core_web_sm'


#*************************** Define methods ******************************************
def user_input_features2(spacy_package):
    model_types = {'Sequential':'sequential', 'Non-sequential':'nonsequential'}
    model_architectures = {'MLP':'mlp', 'CNN':'cnn','LSTM':'lstm'}
    vector_types = {
                    'Word2Vec [Skipgram]':'word2vec_skipgram',
                    'Word2Vec [CBoW]':'word2vec_cbow',
                    'Spacy [' + spacy_package + ']':'spacy',
                    'Tensorflow Keras Tokenizer':'tfkerastokenizer',
                    'TFIDF [Hashing Vectorizer]':'tfidf_hashing_vectorizer'}
    model_optimizers = {'SGD':'sgd', 'Adam':'adam'}
        
    model_type = st.sidebar.selectbox('Model Type', ('Sequential','Non-sequential'))
    model_architecture = st.sidebar.selectbox('Model Architecture', ('MLP','CNN','LSTM'))
    vector_type = st.sidebar.selectbox('Vector Type', (
                                                        'Word2Vec [Skipgram]',
                                                        'Word2Vec [CBoW]',
                                                        'Spacy [' + spacy_package + ']',
                                                        'Tensorflow Keras Tokenizer',
                                                        'TFIDF [Hashing Vectorizer]'))
    model_optimizer = st.sidebar.selectbox('Optimizer', ('SGD','Adam'))
    
    data = {'model_type': model_types[model_type],
            'model_architecture': model_architectures[model_architecture],
            'vector_type': vector_types[vector_type],
            'model_optimizer': model_optimizers[model_optimizer]}
    return pd.DataFrame(data, index=[0])

def user_input_features3(GLOBAL_PATH, model_language, graph_data, graph_data_found, DF_LEN):
    path = GLOBAL_PATH + model_language + '/' + GRAPH_DATA_DIRECTORY + graph_data if graph_data_found else GLOBAL_PATH + 'en/' + GRAPH_DATA_DIRECTORY + GRAPH_DATA_DEFAULT
    metrics = pickle.load(open(path, 'rb'))
    metrics = metrics[[col for col in metrics if not col.startswith('val_')]]
    dataset_size_values = {'en':DF_LEN['en'], 'fi':DF_LEN['fi'], 'sv':DF_LEN['sv'], 'no':DF_LEN['no'], 'de':DF_LEN['de']}

    metric = st.sidebar.selectbox('Metric:', metrics.columns.tolist())
    vector_dimension = st.sidebar.slider('Vector Dimensions', 300)
    dataset_size = st.sidebar.slider('Dataset Size', dataset_size_values[model_language])
    vocabulary_size = st.sidebar.slider('Vocabulary Size', 30000)
    
    data = {'metric': metric,
            'vector_dimension': vector_dimension,
            'dataset_size': dataset_size,
            'vocabulary_size': vocabulary_size}
    return pd.DataFrame(data, index=[0])

def plot_languages(df,model_language):
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size':15})

    col1,col2 = st.columns(2)
    with col1:
        st.write("Language instance count >=")
    with col2:
        #lang_count = st.number_input("",min_value=1,max_value=2000,value=50,label_visibility="collapsed")
        lang_count = st.number_input("",min_value=1,max_value=2000,value=50)

    df_lang = df[df.lang==model_language].lang.value_counts().loc[lambda x: x>=lang_count].to_frame()

    fig = plt.figure(figsize =(10, 7))
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size':15})
    st.write(df_lang)
    plt.pie(df_lang['count'], labels=df_lang.index,autopct='%.2f%%', pctdistance=0.85)
    #https://www.pythonprogramming.in/how-to-pie-chart-with-different-color-themes-in-matplotlib.html
    plt.legend(loc = "best", labels = ['%s = %d' % (l,c) for l,c in zip(df_lang.index, df_lang['count'])]) 
    # plt.show()
    # st.pyplot(fig)

    #https://github.com/streamlit/streamlit/issues/3527
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf, width=650)

def plot_scores(GLOBAL_PATH, model_language, graph_data, user_input_features_df, metric):
    scores = pickle.load(open(GLOBAL_PATH + model_language + '/' + GRAPH_DATA_DIRECTORY + graph_data, 'rb'))
    model_language = user_input_features_df[0].model_language[0]
    model_type = user_input_features_df[1].model_type[0]
    model_architecture = user_input_features_df[1].model_architecture[0]
    model_optimizer = user_input_features_df[1].model_optimizer[0]
    vector_type = user_input_features_df[1].vector_type[0]
    vector_dimension = user_input_features_df[2].vector_dimension[0]
    metric = user_input_features_df[2].metric[0]

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize = (4, 3))
    plt.plot(scores[metric]) 
    plt.plot(scores['val_' + metric]) 
    plt.title(model_type.capitalize() + ' ' +
                model_architecture.upper() + ' ' +
                vector_type + ' ' +
                str(vector_dimension) + 'd ' +
                model_architecture.upper() + ' ' +
                model_optimizer + ' ' +
                metric + ' ' + 
                model_language.upper()) 
    plt.ylabel(metric)
    plt.xlabel('epoch') 
    plt.legend(['train', 'test'], loc = 'best') 
    st.pyplot(fig)

@st.cache_data
def load_dataset(GLOBAL_PATH, model_language):
    return pickle.load(open(GLOBAL_PATH + model_language + '/' + CSV + model_language,'rb'))

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
    
#https://stackoverflow.com/questions/49284455/keras-custom-function-implementing-jaccard
def jaccard_score(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac/100 * smooth

def cleanup_text(GLOBAL_PATH, model_language, spacy_packages, doc):
    nlp = spacy.load(GLOBAL_PATH + model_language + '/spacy/' + spacy_packages[model_language])
    doc = nlp(doc, disable=['parser', 'ner'])
    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
    return ' '.join(tokens)

def print_results(results):
    results_arr = ["Test Loss: {:.2f}%".format(results[0] * 100),
                    "Test Accuracy: {:.2f}%".format(results[1] * 100),
                    "Test Precision: {:.2f}%".format(results[2] * 100),
                    "Test Recall: {:.2f}%".format(results[3] * 100),
                    "Test F-1 Score: {:.2f}%".format(results[4] * 100),
                    "Test Jaccard Score: {:.2f}%".format(results[5] * 100),
                    "Test AUC with ROC graph: {:.2f}%".format(results[6] * 100),
                    "Test Categorical Accuracy: {:.2f}%".format(results[7] * 100),
                    "Test Categorical Crossentropy: {:.2f}%".format(results[8] * 100),
                    "Test Categorical Hinge: {:.2f}%".format(results[9] * 100),
                    "Execution time: {:f}".format(results[len(results)-1])]
    st.write(results_arr)
