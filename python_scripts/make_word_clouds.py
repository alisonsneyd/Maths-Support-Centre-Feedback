# -*- coding: utf-8 -*-
"""
Created on Feb 1 2019

@author: Alison Sneyd

This file takes an input csv with comments to be analysed in a column called 
'comment' and outputs unigram and bigram wordclouds. 

Run'python make_word_clouds.py -h' for help.


"""

# load external modules 
import argparse
import pandas as pd
from sklearn.feature_extraction.text import *
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from utils.lemmatiser import lemmatiser


# set up commandline arguments 
parser = argparse.ArgumentParser()
parser.add_argument(dest='input_file', metavar='FILE', type=str,
                     help='Name of input csv file, data should be in column named comment')
parser.add_argument('-s', dest='stopwords_file',
                     action='append',
                     metavar='FILE',
                     help='Optional text file of additional stopwords')
args = parser.parse_args()


# load data
print('\nLoading data...')
infile = args.input_file
df = pd.read_csv(infile)
print('Input file:', infile)
print('Number comments:', df.shape[0])


# append additional stopwords to sklearns default list if provided
if args.stopwords_file is not None:
    print('Using additional stopwords from:', args.stopwords_file[0])
    with open(args.stopwords_file[0],"r") as stopwords_infile:
        my_stopwords = stopwords_infile.readlines()
        my_stopwords = [word.strip() for word in my_stopwords]
else: 
    my_stopwords = []
Stop_words = ENGLISH_STOP_WORDS.union(my_stopwords)
   


# clean and preprocess data
print('\nCleaning data...')


# function to remove duplicate and very short comments
def clean_comments(df): 
    docs = [comment for comment in df['comment'].unique()]
    docs = [comment for comment in docs if  len(comment.split()) > 10]
    print('Number comments after cleaning:', len(docs))
    return (docs)
docs = clean_comments(df)


# lemmatise comments
lemmatised_docs = lemmatiser(docs)
                                    
                                  
# generate wordclouds
print('\nGenerating unigram and bigram wordclouds...')
     
                  
# function to make unigram wordcloud
def make_unigram_wordcloud(docs): 
    
    # get doc-tf matrix
    # ref https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    vectorizer = CountVectorizer(max_df = 0.50, 
                                 min_df = 5, 
                                 max_features = None,
                                 token_pattern = r'\b[a-zA-Z]{3,}\b',
                                 stop_words=Stop_words)
    doc_tf_vecs = vectorizer.fit_transform(lemmatised_docs)
    doc_tf_words = vectorizer.get_feature_names()
    
    # define term-tf weights dictionaty
    term_weights = {}
    for i in range(doc_tf_vecs.shape[1]):
       term_weights[doc_tf_words[i]] = doc_tf_vecs[:,i].sum()

    # make wordcloud
    # ref https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud.WordCloud
    wordcloud = WordCloud(height = 250, 
                          width = 300,
                          background_color="white",
                          max_font_size=None, 
                          min_font_size=10,
                          random_state = 0,
                          ).generate_from_frequencies(frequencies = term_weights)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    
    # save the generated image
    print('Unigram wordcloud saved to:', infile[:-4]+"_unigram_wordcloud.png")
    plt.savefig(infile[:-4]+"_unigram_wordcloud.png")
make_unigram_wordcloud(lemmatised_docs)
                                  

# function to make bigran wordcloud
def make_bigram_wordcloud(docs): 
    
    # get doc-bigram frequency matrix
    # ref https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    vectorizer = CountVectorizer(max_df = 0.5,
                                 min_df = 5,
                                 token_pattern = r'\b[a-zA-Z]{3,}\b',
                                 stop_words=Stop_words,
                                 ngram_range = (2,2))
    doc_bf_vecs = vectorizer.fit_transform(lemmatised_docs)
    doc_bf_words = vectorizer.get_feature_names()
    
    # define bigram-freq weights dictionary
    term_weights = {}
    for i in range(doc_bf_vecs.shape[1]):
        term_weights[doc_bf_words[i]] = doc_bf_vecs[:,i].sum()
    
    # generate wordcloud
    wordcloud = WordCloud(height = 250, 
                          width = 300,
                          background_color="white",
                          max_font_size=None, 
                          min_font_size=10,
                          random_state = 5,
                          ).generate_from_frequencies(frequencies = term_weights)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    
    # save the generated image
    print('Bigram wordcloud saved to:', infile[:-4]+"_bigram_wordcloud.png")
    plt.savefig(infile[:-4]+"_bigram_wordcloud.png")
make_bigram_wordcloud(lemmatised_docs)

    
                                  
                                  
                            
