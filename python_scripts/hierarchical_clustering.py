# -*- coding: utf-8 -*-
"""
Created on Feb 1 2019

@author: Alison Sneyd

This file takes an input csv with comments to be analysed in a column called 
'comment' and outputs a dendrogram. 

Run'python hierarchical_clustering.py -h' for help.


"""

# load external modules 
import argparse
import pandas as pd
from sklearn.feature_extraction.text import *
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram
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
                                    
                                  
print('\nGenerating dendrogram...')
     
                  
# function to make dendrograms
def make_dendrogram(docs): 
    
    # get doc-tfidf matrix
    # ref https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    vectorizer = TfidfVectorizer(min_df = 10, max_features = 20,  #156
                                token_pattern = r'\b[a-zA-Z]{3,}\b',
                                stop_words=Stop_words,
                                use_idf = True)
    doc_tfidf_vecs = vectorizer.fit_transform(lemmatised_docs)
    tfidf_doc_matrix = doc_tfidf_vecs.transpose()
    doc_tfidf_words = vectorizer.get_feature_names()
    
    # make dendrogram with ward linkage
    # ref:https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
    ward_linkage_matrix = ward(1 - cosine_similarity(tfidf_doc_matrix)) 
    fig, axes = plt.subplots(1,1,figsize=(5, 5))
    axes = dendrogram(ward_linkage_matrix,
                     orientation="right", 
                     labels=doc_tfidf_words)

    # save the generated image
    print('Dendrogram saved to:', infile[:-4]+"_dendrogram.png")
    plt.savefig(infile[:-4]+"_dendrogram.png", bbox_inches = "tight")
make_dendrogram(lemmatised_docs)
                                  


    
                                  
                                  
                            
