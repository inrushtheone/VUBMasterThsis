import warnings
warnings.filterwarnings("ignore")

import os
import re
import numpy as np
import pandas as pd
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
import seaborn as sns

import gensim
from gensim import matutils, utils
from gensim.models import CoherenceModel, LdaModel, TfidfModel, Nmf
from gensim.models.basemodel import BaseTopicModel
from gensim.models.nmf import Nmf as GensimNmf
from gensim.parsing.preprocessing import preprocess_string
from gensim.corpora import Dictionary
from gensim.test.utils import common_texts
import gensim.downloader

from plotnine import ggplot, aes, geom_line

import spacy

import nltk
from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from bokeh.plotting import figure, output_file, show
from bokeh.models import Label, ColumnDataSource, LabelSet
from bokeh.io import output_notebook
from bokeh.layouts import column
from bokeh.models import CustomJS, Slider, Grid, LinearAxis, Plot, Scatter
from bokeh.core.enums import MarkerType

import pickle
from gensim import models

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV','PROPN'], language='NL', notags=True):
    """https://spacy.io/api/annotation"""
    if language == 'NL':
        nlp = spacy.load("nl_core_news_md")
    elif language =='FR':
        nlp = spacy.load("fr_core_news_md")
    elif language == 'EN':
        nlp = spacy.load("en_core_web_lg")
        
    texts_out = []
    texts_out2 = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if (token.pos_ in allowed_postags or notags)])
        
    return texts_out

def remove_stopwords(texts,alltexts,language='NL',num_freq_stopwords=40,extrastopwords=[]):
    
    stop_words_extend = []
    occ,occ_speech = occurence(alltexts)
    occ_speech_rev = inverse_dict(occ_speech)
    for key in sorted(occ_speech_rev)[::-1][0:num_freq_stopwords]:
        for word in occ_speech_rev[key]:
            stop_words_extend.append(word)
    for word in extrastopwords:        
        stop_words_extend.append(word)
    
    if language=='NL':
        stop_words = stopwords.words('dutch')
        stop_words.extend(stop_words_extend)
    else:
        stop_words = stopwords.words('french')
        stop_words.extend(stop_words_extend)
    return [[word for word in doc 
             if word not in stop_words] for doc in texts]


def sent_to_words(sentences,min_len=2):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True, min_len=min_len))    # min_len=2

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV','PROPN'], language='NL', notags=True):
    """https://spacy.io/api/annotation"""
    if language == 'NL':
        nlp = spacy.load("nl_core_news_md")
    elif language =='FR':
        nlp = spacy.load("fr_core_news_md")
        
    texts_out = []
    texts_out2 = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if (token.pos_ in allowed_postags or notags)])
        texts_out2.append([token.lemma_ for token in doc ])
    return texts_out,texts_out2

def occurence(texts):
    occ = {}  
    occ_speech = {}
    for speech in texts:
        for word in speech:
            occ[word] = occ.get(word, 0) + 1
        for word, count in occ.items():
            if word in speech:
                occ_speech[word] = occ_speech.get(word, 0) + 1
    return occ, occ_speech


def counteuh(text):
    count_fre = 0
    for word in text:
        if word in ['euh','euhm']:
            count_fre += 1
    return count_fre
        

def inverse_dict(input_dict):
    inverse = {}
    for key, value in input_dict.items():
        token = inverse.get(value, [])
        token.append(key)
        inverse[value] = token
        
    return inverse

def compute_coherence_values(dictionary, corpus, texts, limit=40, start=2, step=2):

    
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary, num_topics=num_topics,random_state=88)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score - LDA")
    plt.ylim(0.1,0.7)
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    
    coherence_values1 = []
    model_list = []
    for num_topics in range(start, limit, step):
        
        model = Nmf(corpus=corpus,id2word=dictionary, num_topics=num_topics,random_state=88)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values1.append(coherencemodel.get_coherence())
        
    x = range(start, limit, step)
    plt.plot(x, coherence_values1)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score - NMF")
    plt.ylim(0.1,0.7)
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    

    return coherence_values,coherence_values1

def compute_coherence_values_random(dictionary, corpus, texts, limit=20, start=0, step=1):
    num_topics=10

    coherence_values = []
    model_list = []
    for random_state in range(start, limit, step):
        
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary, num_topics=num_topics,random_state=random_state)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("random state value")
    plt.ylabel("Coherence score - LDA")
    plt.ylim(0.1,0.7)
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    
    coherence_values = []
    model_list = []
    for random_state in range(start, limit, step):
        
        model = Nmf(corpus=corpus,id2word=dictionary, num_topics=num_topics,random_state=random_state)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("random state value")
    plt.ylabel("Coherence score - NMF")
    plt.ylim(0.1,0.7)
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    

    return None

def removedwords(texts1,texts2):
    set1 = []
    set2 = []
    for text in texts1:
        for word in text:
            set1.append(word)
    for text in texts2:
        for word in text:
            set2.append(word)   
            
    set1 = set(set1)
    set2 = set(set2)
    print('the following tokens are removed')
    print(set1 - set2)
    return None

def pre_processing(speeches,input_type='filename', language='NL', 
                   min_len=2, lem=True, notags=False,allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV','PROPN'],num_freq_stopwords=40,extrastopwords=[]):
    
    #The input is either a filename, or a dataframe serie
    if input_type=='filename':
        df = pd.read_csv(speeches)
        data = df['0']
    else:
        data = speeches
        
    #Remove punctuation 
    data = data.map(lambda x: re.sub('[,\\.!?]', '', x))
    
    #Convert the text to lowercase
    data = data.map(lambda x: x.lower())
    
    #text to words 
    data_words = data.map(lambda x: x.split())
    
    original = data
    #lemmatize
    if lem:
        lemma = lemmatization(data_words, language=language, allowed_postags=allowed_postags, notags=notags)
        data_words_lemm = lemma[0]
    else:
        data_words_lemm = data_words
    
    #remove stopwords
    
    data_words_lemm_stopwordsremoved = remove_stopwords(data_words_lemm,
                                                        language=language,
                                                        num_freq_stopwords=num_freq_stopwords,
                                                        extrastopwords=extrastopwords,
                                                       alltexts=lemma[1])    
    id2word = corpora.Dictionary(data_words_lemm_stopwordsremoved)
    corpus = [id2word.doc2bow(text) for text in data_words_lemm_stopwordsremoved]
    
    return id2word,corpus,data_words_lemm_stopwordsremoved,original,data_words_lemm,lemma[1]
        
def createmodels(id2word,corpus,num_topics=5):
    
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word, num_topics=num_topics,random_state=88)
    nmf_model = Nmf(corpus=corpus, id2word=id2word,num_topics=num_topics,random_state=88)
    
    tfidf = TfidfModel(dictionary=id2word)
    corpus_tfidf = list(tfidf[corpus])
    nmf_tfidf_model = Nmf(corpus=corpus_tfidf, id2word=id2word,num_topics=num_topics,random_state=88)
    
    return lda_model, nmf_model, nmf_tfidf_model

def overview(num_topics=10, num_freq_stopwords=40, language='NL', notags=False, min_len=2,filename='Dutch.csv',extrastopwords=[]):
    print(f"The number of topics is {num_topics}." )
    print(f"Tokens with the {num_freq_stopwords} highest document frequency are added to the stopwords and removed from the corpus.")
    print(f'All the selected tokens have a minimum length of {min_len}.')    
    data = pre_processing(filename,min_len=min_len,notags=True,extrastopwords=extrastopwords,language=language)    
    corpus = data[1]
    id2word = data[0] 
    input_tokens = data[2]
    texts = data[3]
    
    print('The descriptive statistics of the processed data is shown below.')
    descriptive_data(input_tokens, n1=num_freq_stopwords)
    
    removedwords(data[5],input_tokens)
    
    return None

def descriptive_data(df_input,n1=20):
    occ, occ_speech = occurence(df_input)
    occ_rev = inverse_dict(occ)
    occ_speech_rev = inverse_dict(occ_speech)
    
    print('The tokens with the highest document frequence are')
    for key in sorted(occ_speech_rev)[::-1][0:n1]:        
        print(occ_speech_rev[key])
        
    print('The tokens with the highest term frequence are')   
    for key in sorted(occ_rev)[::-1][0:n1]:        
        print(occ_rev[key])
    print('The descriptive statistics for the term frequence are presented below:')
    print(pd.DataFrame(list(occ.values())).describe())
    print('The descriptive statistics for the document frequence are presented below:')
    print(pd.DataFrame(list(occ_speech.values())).describe())
    
    return None 
    
    
def readdata(filename,language='NL'):
    NL = pd.read_csv(filename)
    # Remove punctuation
    NL['text_processed'] = NL['0'].map(lambda x: re.sub('[,\\.!?]', '', x))
    # Convert the titles to lowercase
    NL['text_processed'] = NL['text_processed'].map(lambda x: x.lower())
    # Print out the first rows 
    NL['text_word'] = NL['text_processed'].map(lambda x: x.split())
    NL['text_length'] = NL['text_processed'].map(lambda x: len(x.split()))
    NL['word_lem'] = lemmatization(NL['text_word'], notags=True,language=language)[0]
    NL['text_length_lem'] = NL['word_lem'].map(lambda x: len(x))
    NL['word_uni'] = NL['text_word'].map(lambda x:set(x))
    
    return NL

def format_topics_sentences(model, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    sent_topics_df = sent_topics_df.reset_index()
    sent_topics_df.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    return(sent_topics_df)

def plotwordcloud(model):
    for t in range(model.num_topics):
        plt.figure(figsize=(4,3),dpi=400)
        word_freq = {}
        for word in model.show_topic(t, 100):
            word_freq[word[0]] = word[1]

        plt.imshow(WordCloud(background_color='white').fit_words(word_freq))
        plt.axis("off")
        plt.title("Topic #" + str(t))
        plt.show()
    return None

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

def plot_wordcloud(model, folderpath=None):
    """
    Plots a word cloud for each topic in the given topic model.
    
    Parameters:
    model (gensim model): The topic model to visualize.
    folderpath (str, optional): The directory to save the word cloud images. If not provided, the images are not saved.
    """
    for t in range(model.num_topics):
        plt.figure(figsize=(8, 6), dpi=300)
        word_freq = {word: weight for word, weight in model.show_topic(t, topn=100)}
        
        wordcloud = WordCloud(
            background_color='white',
            width=1800,
            height=2400,
            max_words=100,
            colormap='gray'  # Using grayscale colormap for APA style
        ).fit_words(word_freq)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Topic {t}", fontsize=12, fontname='Arial')
        plt.tight_layout()
        
        if folderpath:
            if not os.path.exists(folderpath):
                os.makedirs(folderpath)
            plt.savefig(os.path.join(folderpath, f"Topic_{t}.png"), bbox_inches='tight')
        else:
            plt.show()

    return None


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from collections import Counter
import os

def plotbar(model, texts, folderpath=None):
    """
    This function visualizes the term frequency and word weight of keywords for each topic in a given topic model.
    It first extracts the number of topics and the corresponding keywords from the model. Then, it flattens the
    provided text data to count the frequency of each word. Using this data, it constructs a DataFrame that includes
    the word, its associated topic, word weight, and word count. The function then identifies the global maximum
    values for term frequency and word weight across all topics to set uniform y-axis limits for easier comparison. It
    creates a series of bar plots using Matplotlib, where each subplot represents a topic with term frequency and 
    word weight bars. The plots share consistent y-axis limits, ensuring a coherent comparison across topics.

    Parameters:
    model (gensim model): The topic model to visualize.
    texts (list of list of str): The processed text data.
    folderpath (str, optional): The directory to save the bar plots. If not provided, the plots are displayed.
    """

    # Number of topics
    num_topics = model.num_topics
    topics = model.show_topics(formatted=False, num_topics=num_topics)
    
    # Flatten text data
    data_flat = [word for sublist in texts for word in sublist]
    word_counter = Counter(data_flat)

    # Prepare data for plotting
    plot_data = []
    for topic_id, topic in topics:
        for word, weight in topic:
            plot_data.append([word, topic_id, weight, word_counter[word]])

    df = pd.DataFrame(plot_data, columns=['word', 'topic_id', 'word weight', 'word_count'])

    # Find global maximum values for word_count and word weight
    global_max_word_count = df['word_count'].max() * 1.1
    global_max_word_weight = df['word weight'].max() * 1.1

    # Use more visually distinct colors
    colors = ['#708090', '#C0C0C0']  # Slate Gray and Silver for contrast

    for i in range(num_topics):
        fig, ax = plt.subplots(figsize=(6.4, 4.5), dpi=300)  # Maintain size from your example

        topic_data = df[df.topic_id == i]
        ax.bar(x=topic_data['word'], height=topic_data['word_count'], color=colors[0], width=0.4, alpha=0.8, label='Term Frequency')
        ax_twin = ax.twinx()
        ax_twin.bar(x=topic_data['word'], height=topic_data['word weight'], color=colors[1], width=0.4, alpha=0.6, label='Word Weight', align='edge')

        ax.set_ylabel('Total Term Frequency', fontsize=12, fontname='Arial', color='gray')
        ax_twin.set_ylabel('Word Weight', fontsize=12, fontname='Arial', color='gray')
        
        # Set y-limits to the global maximum values
        ax.set_ylim(0, global_max_word_count)
        ax_twin.set_ylim(0, global_max_word_weight)
        
        ax.set_title(f'Topic {i}', fontsize=12, fontname='Arial', color='darkslategray')
        ax.tick_params(axis='x', rotation=45, labelsize=10, labelcolor='darkslategray')
        ax.tick_params(axis='y', labelsize=10, labelcolor='darkslategray')
        ax_twin.tick_params(axis='y', labelsize=10, labelcolor='darkslategray')
        ax.set_xticklabels(topic_data['word'], fontname='Arial')

        # Explicitly setting legend font to Arial
        legend = ax.legend(loc='upper left', fontsize=10)
        for text in legend.get_texts():
            text.set_fontname('Arial')
        legend_twin = ax_twin.legend(loc='upper right', fontsize=10)
        for text in legend_twin.get_texts():
            text.set_fontname('Arial')

        fig.tight_layout()

        if folderpath:
            if not os.path.exists(folderpath):
                os.makedirs(folderpath)
            fig.savefig(os.path.join(folderpath, f'barplot_topic_{i}.png'), bbox_inches='tight')
        else:
            plt.show()
        plt.close(fig)  # Close the figure after saving or displaying

    return None

# Example usage:
# plotbar(lda_model, texts, "output_folder")

def topics_per_document(model, corpus, start=0, end=1):
    corpus_sel = corpus
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs = model[corp]
        
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)
    return dominant_topics,topic_percentages

def plottopicdocument(model,corpus):
    dominant_topics, topic_percentages = topics_per_document(model=model, corpus=corpus, end=-1)            
    # Distribution of Dominant Topics in Each Document
    df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
    dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
    df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

    # Total Topic Distribution by actual weight
    topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
    df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

    # Top 3 Keywords for each Topic
    num_topics = model.num_topics
    topic_top3words = [(i, topic) for i, topics in model.show_topics(formatted=False,num_topics=num_topics) 
                                     for j, (topic, wt) in enumerate(topics) if j < 3]

    df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
    df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
    df_top3words.reset_index(level=0,inplace=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(num_topics*2.5, 10), dpi=240, sharey=True)

    # Topic Distribution by Dominant Topics
    ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
    ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
    tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
    ax1.xaxis.set_major_formatter(tick_formatter)
    ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=25))
    ax1.set_ylabel('Number of Documents')
    

    # Topic Distribution by Topic Weights
    ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')
    ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
    ax2.xaxis.set_major_formatter(tick_formatter)
    ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=25))
    
    return None 
    
    


def printtopics(model,n=10):
    df = pd.DataFrame()
    num_topics = model.num_topics
    
    for i in range(num_topics):
        wp = model.show_topic(i,n)
        topic_keywords = ", ".join([word for word, prop in wp])
        df = df.append(pd.Series([int(i),  topic_keywords]), ignore_index=True)
    
    return df 


def keywordsweight(model):
    nt = model.num_topics
    n1 = 1002
    df = pd.DataFrame()
    df['top n'] = [5,10,20,50,100,200,500,1000]
    for t in range(nt):
        
        total= 0
        per = []
        for w in range(n1):
            total +=model.show_topic(t, n1)[w][1]
            if (w+1) in list(df['top n']):
                per.append(round(total,3))
                
        df[t] = per
    return df


def top10(model,n=10):
    
    nt = model.num_topics
    df = pd.DataFrame()
    df['Keywords number'] = range(n)

    for t in range(nt):
        words = []
        freq = []
        for word in model.show_topic(t, n):
            words.append(word[0])
            freq.append(word[1])
        df['topic' + str(t) + ' token'] = words
        df['topic' + str(t) + ' weight'] = freq
        
    return df


def topicdistribution(model,corpus,filename=None):
    
    topic_weights = []
    num_topics = model.num_topics
    for i, row_list in enumerate(model[corpus]):
        tw = [0]*num_topics
        for j,p in row_list:
            tw[j] = p
        topic_weights.append(tw)

    arr = pd.DataFrame(topic_weights).fillna(0).values
    tsne_model = TSNE(n_components=2, verbose=1, random_state=88, angle=.99, init='pca',n_iter=1000)
    tsne = tsne_model.fit_transform(arr)

    toR = pd.DataFrame()
    toR['x_tsne'] = tsne[:,0]
    toR['y_tsne'] = tsne[:,1]
    toR['x_1_topic_probability'] = np.amax(arr, axis=1)
    for t in range(num_topics):
        name = 'topic' + str(t+1)
        toR[name] = arr[:,t]
    if filename is not None:
        toR.to_csv(filename,sep=";",encoding = "ISO-8859-1")
    return toR




def createtsne(model,corpus):
    topic_weights = []
    num_topics = model.num_topics
    for i, row_list in enumerate(model[corpus]):
        tw = [0]*num_topics
        for j,p in row_list:
            tw[j] = p
        topic_weights.append(tw)

    # Array of topic weights    
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # Keep the well separated points (optional)
    #arr = arr[np.amax(arr, axis=1) > 0.35]

    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_outcome = tsne_model.fit_transform(arr)
    
    return tsne_outcome, topic_num



    
    
import numpy as np
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, LabelSet

def showtsne(model0, corpus0, legendtype='color', legend=False):
    """
    Function to visualize t-SNE clustering using Bokeh.
    
    Parameters:
    model0 : LDA model
        The trained LDA model.
    corpus0 : list
        The corpus on which the LDA model was trained.
    legendtype : str, optional
        Type of legend to use ('color' or 'marker'). Default is 'color'.
    legend : bool, optional
        Whether to display the legend. Default is False.
    
    Returns:
    None
    """
    
    # Generate t-SNE and topic numbers
    tsne0, topic_num0 = createtsne(model0, corpus0)

    # Define colors and markers for 20 categories
    mycolors = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                         '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#00FFFF', '#592720',
                         '#9b59b6', '#e74c3c', '#34495e', '#2ecc71', '#e67e22', '#f1c40f',
                         '#e84393', '#3498db'])
    markers = np.array(['asterisk', 'circle', 'cross', 'dash', 'diamond', 'dot', 'hex', 'inverted_triangle',
                        'plus', 'square', 'triangle', 'x', 'diamond_cross', 'diamond_dot', 
                        'circle_cross', 'circle_dot', 'circle_x', 'dash_dot', 'dash_x', 'triangle_dot'])
    labels = np.array(topic_num0)

    # Create a ColumnDataSource for the plot
    source = ColumnDataSource(data=dict(
        x0=tsne0[:, 0],
        y0=tsne0[:, 1],
        x=tsne0[:, 0],
        y=tsne0[:, 1],
        colour=mycolors[topic_num0 % 20],  # Modulo to ensure indices are within range
        markers=markers[topic_num0 % 20],
        label=labels,
        names=list(range(len(corpus0)))  # Adjusted to the length of the corpus
    ))

    # Initialize the plot
    plot = figure(title="t-SNE Clustering", plot_width=900, plot_height=700)

    # Scatter plot with color legend
    if legendtype == 'color':
        plot.scatter(x='x', y='y', source=source, legend_field='label', color='colour', size=20)
    # Scatter plot with marker legend
    else:
        plot.scatter(x='x', y='y', source=source, legend_field='label', size=20, marker='markers')
    
    # Add legend if specified
    if legend:
        plot.legend.orientation = "horizontal"

    # Add labels if legend is specified
    if legend:
        labels = LabelSet(x='x', y='y', text='names', x_offset=5, y_offset=5, source=source, render_mode='canvas')
        plot.add_layout(labels)
    
    plot.legend.title = "Dominant topic"

    # Show the plot
    show(plot)

    return None

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


from IPython.display import display, HTML

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
from IPython.display import display, HTML

def plottopicdocument2(model, corpus, folderpath=None):
    """
    This function visualizes the distribution of topics in a set of documents using a given topic model. 
    It plots two main aspects:
    1. The number of documents dominated by each topic.
    2. The total weightage of each topic across all documents.
    It also generates an APA-style table listing the top ten words for each topic along with their weights.

    Parameters:
    model (gensim model): The topic model to visualize.
    corpus (list of list of (int, int)): The corpus in BOW format.
    folderpath (str, optional): The directory to save the plots. If not provided, the plots are displayed.
    """
    # Extract dominant topics and topic percentages
    dominant_topics, topic_percentages = topics_per_document(model=model, corpus=corpus)
    
    # Distribution of Dominant Topics in Each Document
    df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
    dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
    df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()
    
    # Total Topic Distribution by actual weight
    topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
    df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()
    
    # Top 10 Keywords for each Topic with their weights
    num_topics = model.num_topics
    topic_top10words = [(i, topic, wt) for i, topics in model.show_topics(formatted=False, num_topics=num_topics, num_words=10) 
                        for j, (topic, wt) in enumerate(topics) if j < 10]
    
    df_top10words_stacked = pd.DataFrame(topic_top10words, columns=['topic_id', 'words', 'weight'])
    df_top10words_stacked['words_with_weights'] = df_top10words_stacked.apply(lambda x: f"{x['words']}({x['weight']:.4f})", axis=1)
    df_top10words = df_top10words_stacked.groupby('topic_id').agg({
        'words_with_weights': ', '.join,
        'weight': 'sum'
    }).reset_index()
    
    # Increase the figure size for better readability
    fig, ax1 = plt.subplots(figsize=(6.4, 4.5))  # Adjust the width (6.4) and height (4.5) as needed

    # Create a bar plot for the distribution of dominant topics in each document
    ax1.bar(df_dominant_topic_in_each_doc['Dominant_Topic'], df_dominant_topic_in_each_doc['count'], 
            color='gray', edgecolor='black', width=0.5, label='Number of Documents')

    # Set the axis labels for the histogram
    ax1.set_xlabel('Topics', fontsize=12, fontname='Arial')
    ax1.set_ylabel('Number of Documents', fontsize=12, fontname='Arial')

    # Set the tick label font size and style for the histogram
    ax1.tick_params(axis='x', labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    plt.xticks(range(19), fontsize=10, fontname='Arial')
    plt.yticks(fontsize=10, fontname='Arial')

    # Remove the top and right spines for a cleaner look
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Create a second y-axis for the total topic weightage
    ax2 = ax1.twinx()
    ax2.bar(df_topic_weightage_by_doc['index'], df_topic_weightage_by_doc['count'], 
            color='black', alpha=0.6, width=0.2, label='Total Topic Weight')

    # Set the y-axis label for the total topic weightage
    ax2.set_ylabel('Total Topic Weight', fontsize=12, fontname='Arial')

    # Set the tick label font size and style for the scatter line
    ax2.tick_params(axis='y', labelsize=10)
    plt.yticks(fontsize=10, fontname='Arial')

    # Move the legend to the top center and remove the border line
    legend = fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1), bbox_transform=ax1.transAxes, frameon=False)

    # Set the legend font size and style
    for text in legend.get_texts():
        text.set_fontsize(10)
        text.set_fontname('Arial')

    # Remove grid lines for a cleaner appearance
    ax1.grid(False)
    ax2.grid(False)

    # Set uniform y-axis limits
    max_height = max(df_dominant_topic_in_each_doc['count'].max(), df_topic_weightage_by_doc['count'].max()) * 1.1
    ax1.set_ylim(0, max_height)
    ax2.set_ylim(0, max_height)

    plt.tight_layout()

    # Save or display the plot
    if folderpath:
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        plt.savefig(os.path.join(folderpath, 'topic_document_distribution.png'), dpi=1600, bbox_inches='tight')
        # Save the table to a CSV file
        df_top10words.to_csv(os.path.join(folderpath, 'top_ten_words_with_weights.csv'), index=False)
        plt.show()
    else:
        plt.show()

    # Close the figure after saving or displaying
    plt.close(fig)

    # Generate APA-style table for top ten words per topic with their weights
    table_html = df_top10words[['topic_id', 'words_with_weights', 'weight']].to_html(index=False, justify='center')
    display(HTML(f"""
    <style>
        table {{
            width: 70%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 16px;
            text-align: center;
        }}
        table, th, td {{
            border: 1px solid black;
        }}
        th, td {{
            padding: 12px 15px;
        }}
        thead {{
            background-color: #f2f2f2;
        }}
    </style>
    <h2>Table 1</h2>
    <p>Top Ten Words for Each Topic with Weights</p>
    {table_html}
    """))

    return None





import random
def plottsne(model, corpus):
    """
    This function performs t-SNE dimensionality reduction on topic model weights and visualizes the resulting clusters
    using Bokeh. Each document is represented as a point colored by its dominant topic. Labels are randomly assigned to 200 documents.
    """
    # Get topic weights for each document
    topic_weights = []
    num_topics = model.num_topics
    for i, row_list in enumerate(model[corpus]):
        tw = [0] * num_topics
        for j, p in row_list:
            tw[j] = p
        topic_weights.append(tw)
    
    # Convert topic weights to a NumPy array
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # Find the dominant topic for each document
    topic_num = np.argmax(arr, axis=1)

    # Perform t-SNE dimensionality reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=0.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    # Bokeh plot setup
    output_notebook()
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    plot = figure(title=f"t-SNE Clustering of {num_topics} Topics",
                  plot_width=900, plot_height=700)
    plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=mycolors[topic_num % len(mycolors)])

    # Randomly select 200 documents to label
    total_docs = len(tsne_lda)
    if total_docs > 200:
        selected_indices = random.sample(range(total_docs), 200)
    else:
        selected_indices = range(total_docs)

    selected_x = tsne_lda[selected_indices, 0]
    selected_y = tsne_lda[selected_indices, 1]
    selected_names = [f"Doc {i}" for i in selected_indices]

    # Add labels to the plot
    source = ColumnDataSource(data=dict(
        x=selected_x,
        y=selected_y,
        names=selected_names
    ))
    labels = LabelSet(x='x', y='y', text='names',
                      x_offset=5, y_offset=5, source=source, render_mode='canvas')
    plot.add_layout(labels)
    
    # Show the plot
    show(plot)
    
    return None




def plot_citation_percentiles(lda_model, corpus, n_citation, color='viridis', file=''):
    # Ensure the input is a DataFrame
    citation = pd.DataFrame({'n_citation': n_citation})
    
    # Get the t-SNE outcome and topic numbers
    tsne_outcome, topic_num = createtsne(lda_model, corpus)

    # Extract x and y coordinates from the t-SNE result
    x_coordinates = tsne_outcome[:, 0]
    y_coordinates = tsne_outcome[:, 1]
    
    # Calculate the percentiles of 'n_citation'
    citation['citation_percentile'] = citation['n_citation'].rank(pct=True)
    
    # Determine the colormap based on the color input
    cmap = 'gray' if color == 'gray' else 'viridis'
    
    # Create a scatter plot with 'citation_percentile' values determining point colors
    plt.figure(figsize=(10, 10))  # A4 size in inches (portrait orientation)
    
    # Scatter plot with 'x_coordinates' as x-coordinates, 'y_coordinates' as y-coordinates, and 'citation_percentile' as colors (cmap)
    scatter = plt.scatter(x_coordinates, y_coordinates, c=citation['citation_percentile'], cmap=cmap, s=100)
    
    # Add color bar for reference
    cbar = plt.colorbar(scatter)
    cbar.set_label('Citation Percentile', fontsize=14, fontname='Times New Roman')
    
    # Label the axes and title in Times New Roman for APA style
    plt.xlabel('t-sne dimension 1', fontsize=14, fontname='Times New Roman', color='black')
    plt.ylabel('t-sne dimension 2', fontsize=14, fontname='Times New Roman', color='black')
    plt.title('', fontsize=16, fontname='Times New Roman', color='black')
    
    # Customize tick parameters
    plt.tick_params(axis='both', which='major', labelsize=12, labelcolor='black')
    
    # Set font properties for APA style
    plt.xticks(fontsize=12, fontname='Times New Roman')
    plt.yticks(fontsize=12, fontname='Times New Roman')
    
    # Remove gridlines
    plt.grid(False)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save the plot if a file name is provided
    if file:
        plt.savefig(file, format='png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()