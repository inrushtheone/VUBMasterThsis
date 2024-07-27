# nltk.download('stopwords')
#!python -m spacy download en_core_web_lg

import warnings
warnings.filterwarnings("ignore")
import json
import csv
import os
import re
import numpy as np
import pandas as pd
from collections import Counter
import spacy
from gensim import corpora

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

def combinecsv(file1, file2, file3=None, sep1=",", sep2=","):
    # Read the CSV files without importing the row number as a column
    df1 = pd.read_csv(file1, sep=sep1, index_col=False)
    df2 = pd.read_csv(file2, sep=sep2, index_col=False)
    
    # Check if they have the same number of rows
    if df1.shape[0] != df2.shape[0]:
        raise ValueError("The two files do not have the same number of rows.")
    
    # Combine the DataFrames column-wise
    combined_df = pd.concat([df1, df2], axis=1)
    
    # If file3 is provided, save the combined DataFrame to that file
    if file3 is not None:
        combined_df.to_csv(file3, index=False)
    
    return combined_df

def topicdistribution(model, corpus, filename=None):
    topic_weights = []
    num_topics = model.num_topics
    for i, row_list in enumerate(model[corpus]):
        tw = [0] * num_topics
        for j, p in row_list:
            tw[j] = p
        topic_weights.append(tw)
    
    arr = pd.DataFrame(topic_weights).fillna(0).values
    tsne_model = TSNE(n_components=2, random_state=0, verbose=1,angle=0.99, init='pca')
    tsne = tsne_model.fit_transform(arr)

    toR = pd.DataFrame()
    toR['x_tsne'] = tsne[:, 0]
    toR['y_tsne'] = tsne[:, 1]
    toR['x_1_topic_probability'] = np.amax(arr, axis=1)
    toR['dominant_topic'] = np.argmax(arr, axis=1) 
    for t in range(num_topics):
        name = 'topic' + str(t)
        toR[name] = arr[:, t]

    if filename is not None:
        toR.to_csv(filename, sep=";", encoding="ISO-8859-1")

    return toR

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



def plotbar(model,texts):


    num_topics = model.num_topics
    topics = model.show_topics(formatted=False,num_topics=num_topics)
    data_ready = texts
    data_flat = [w for w_list in data_ready for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

    plotsize = num_topics//2
    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(plotsize, 2, figsize=(num_topics,num_topics*1.3), sharey=True, dpi=320)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i%10], width=0.5, alpha=0.3, label='Term frequency')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i%10], width=0.2, label='Weights')
        if i%2==0:
            ax.set_ylabel('Total term frequency', color=cols[i%10])
        else:
            ax_twin.set_ylabel('Weight', color=cols[i%10])
        ax_twin.set_ylim(0, 0.020); ax.set_ylim(0, 100)
        ax.set_title('Topic: ' + str(i), color=cols[i%10], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
        ax.legend(loc='upper left',bbox_to_anchor=(-0.1,1.25)); ax_twin.legend(loc='upper right',bbox_to_anchor=(1,1.25))

    fig.tight_layout(w_pad=0)    
    #fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
    plt.show()
    
    return None

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
    
    
def plottsne(model,corpus):

    # Get topic weights
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
    tsne_lda = tsne_model.fit_transform(arr)


    # Plot the Topic Clusters using Bokeh
    output_notebook()
    n_topics = num_topics
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    plot = figure(title="t-SNE Clustering of {} Topics".format(n_topics), 
                  plot_width=900, plot_height=700)
    plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num%10])
    

    source = ColumnDataSource(data=dict(
        x=tsne_lda[:,0],
        y=tsne_lda[:,1],
        names=list(range(108))))
    
    



    labels = LabelSet(x='x', y='y', text='names', 
                  x_offset=5, y_offset=5, source=source, render_mode='canvas')
    plot.add_layout(labels)
    show(plot)
    
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



def plottopicdocument2(model,corpus):
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
    df_top3words = df_top3words_stacked.groupby('topic_id').agg('\n'.join)
    df_top3words.reset_index(level=0,inplace=True)
    
    fig, (ax1) = plt.subplots(1, 1, figsize=(14, 6), dpi=240, sharey=True)

    # Topic Distribution by Dominant Topics
    ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick',alpha=0.3,label='by dominant topic')
    ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
    tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
    ax1.xaxis.set_major_formatter(tick_formatter)
    #ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=25))
    ax1.set_ylabel('Count of documents')
    ax_twin = ax1.twinx()
    ax_twin.bar(x='index', height="count", data=df_topic_weightage_by_doc, color='steelblue', width=0.2, label='by sum of weights')
    ax1.legend(loc='upper left'); ax_twin.legend(loc='upper right')
    ax_twin.set_ylim(0, 30); ax1.set_ylim(0, 30)
    
    
    
    
    return None 







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


def comparetsne(model0,corpus0,model1,corpus1):
    
    tsne0,topic_num0 = createtsne(model0,corpus0)
    tsne1,topic_num1 = createtsne(model1,corpus1)

    mycolors = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
       '#e377c2', '#7f7f7f', '#bcbd22', '#17becf','#00FFFF','#592720'])
    markers = np.array(['asterisk', 'circle', 'cross', 'dash', 'diamond', 'dot', 'hex',  'inverted_triangle',
           'plus', 'square', 
            'triangle', 'x' ])
    labels = np.array(topic_num1)

    source = ColumnDataSource(data=dict(
        x0=tsne0[:,0],
        y0=tsne0[:,1],
        x1=tsne1[:,0],
        y1=tsne1[:,1],
        x=tsne0[:,0],
        y=tsne0[:,1],
        colour=mycolors[topic_num0],
        markers=markers[topic_num1],
        label=labels,
        names=list(range(108))))
    # Plot the Topic Clusters using Bokeh
    #output_notebook()
    n_topics = num_topics

    
    plot = figure(title="t-SNE Clustering ", 
                  plot_width=900, plot_height=700)



    #glyph = Scatter(x="x", y="y")
    #plot.add_glyph(source, glyph)
    plot.scatter(x='x', y='y', source=source, legend_group='label',color='colour',size=20,marker='markers')
    #plot.legend.orientation = "horizontal"




    labels = LabelSet(x='x', y='y', text='names', 
                  x_offset=5, y_offset=5, source=source, render_mode='canvas')
    slider = Slider(start=0, end=1, value=0, step=.1, title="model selection")


    update_curve = CustomJS(args=dict(source=source, slider=slider), code="""
        var data = source.data;
        var f = 1-slider.value;
        var x = data['x']
        var y = data['y']
        var x0 = data['x0']
        var y0 = data['y0']
        var x1 = data['x1']
        var y1 = data['y1']
        for (var i = 0; i < x.length; i++) {
            y[i] = f*y0[i] + (1-f)*y1[i]
            x[i] = f*x0[i] + (1-f)*x1[i]

        }

        // necessary becasue we mutated source.data in-place
        source.change.emit();
    """)
    slider.js_on_change('value', update_curve)

    plot.add_layout(labels)



    show(column(slider, plot))
    
    return None
    
    
    
    
def showtsne(model0,corpus0,legendtype='color',legend=False):
    
    tsne0,topic_num0 = createtsne(model0,corpus0)
    

    mycolors = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
       '#e377c2', '#7f7f7f', '#bcbd22', '#17becf','#00FFFF','#592720'])
    markers = np.array(['asterisk', 'circle', 'cross', 'dash', 'diamond', 'dot', 'hex',  'inverted_triangle',
           'plus', 'square', 
            'triangle', 'x' ])
    labels = np.array(topic_num0)

    source = ColumnDataSource(data=dict(
        x0=tsne0[:,0],
        y0=tsne0[:,1],
        
        x=tsne0[:,0],
        y=tsne0[:,1],
        colour=mycolors[topic_num0],
        markers=markers[topic_num0],
        label=labels,
        names=list(range(108))))
    # Plot the Topic Clusters using Bokeh
    #output_notebook()
    

    
    plot = figure(title="t-SNE Clustering ", 
                  plot_width=900, plot_height=700)



    #glyph = Scatter(x="x", y="y")
    #plot.add_glyph(source, glyph)
    #plot.scatter(x='x', y='y', source=source, legend_group='label',color='colour',size=20,marker='markers')
    if legendtype=='color':
        plot.scatter(x='x', y='y', source=source, legend_group='label',color='colour',size=20)
    else:
        plot.scatter(x='x', y='y', source=source, legend_group='label',size=20,marker='markers')
    if legend:
        plot.legend.orientation = "horizontal"




    labels = LabelSet(x='x', y='y', text='names', 
                  x_offset=5, y_offset=5, source=source, render_mode='canvas')
    
    if legend:
        plot.add_layout(labels)
    plot.legend.title = "Dominant topic"



    show(plot)
    
    return None 

def plottopicdocument2(model, corpus):
    """
    This function visualizes the distribution of topics in a set of documents using a given topic model. 
    It plots two main aspects:
    1. The number of documents dominated by each topic.
    2. The total weightage of each topic across all documents.
    """
    # Extract dominant topics and topic percentages
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
    topic_top3words = [(i, topic) for i, topics in model.show_topics(formatted=False, num_topics=num_topics) 
                       for j, (topic, wt) in enumerate(topics) if j < 3]
    
    df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
    df_top3words = df_top3words_stacked.groupby('topic_id').agg('\n'.join)
    df_top3words.reset_index(level=0, inplace=True)
    
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 6), dpi=240, sharey=True)
    
    # Topic Distribution by Dominant Topics
    ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=0.5, color='firebrick', alpha=0.3, label='by dominant topic')
    ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
    tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(int(x)) + '\n' + df_top3words.loc[df_top3words.topic_id == int(x), 'words'].values[0])
    ax1.xaxis.set_major_formatter(tick_formatter)
    ax1.set_ylabel('Count of documents')
    ax1.legend(loc='upper left')
    
    # Total Topic Distribution by Weights
    ax_twin = ax1.twinx()
    ax_twin.bar(x='index', height="count", data=df_topic_weightage_by_doc, color='steelblue', width=0.2, label='by sum of weights')
    ax_twin.set_ylabel('Sum of topic weights')
    ax_twin.legend(loc='upper right')
    
    # Set uniform y-axis limits
    max_height = max(df_dominant_topic_in_each_doc['count'].max(), df_topic_weightage_by_doc['count'].max()) * 1.1
    ax1.set_ylim(0, max_height)
    ax_twin.set_ylim(0, max_height)
    
    plt.tight_layout()
    plt.show()

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