# -*- coding: utf-8 -*-
"""
Created on Feb 1 2019

@author: Alison Sneyd


This file takes an input csv with comments to be analysed in a column called 
'comment' and implements LDA. The output is a text file of topics, represented 
by their most probable words and comments.

Run'python topic_modelling.py -h' for help.

"""

# load external modules 
import argparse
import pandas as pd
from nltk.stem import WordNetLemmatizer 
from nltk import word_tokenize
from sklearn.feature_extraction.text import *
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np



# set up commandline arguments 
parser = argparse.ArgumentParser()
parser.add_argument(dest='input_file', metavar='FILE', type=str,
                     help='Name of input file, should be csv with data in column named comment')
parser.add_argument('-t', dest='number_topics',
                     action='append',
                     help='Option to set number of topics, default is 20')
parser.add_argument('-w', dest='number_words',
                     action='append',
                     help='Option to set number of topic words to output, default is 10')
parser.add_argument('-d', dest='number_docs',
                     action='append',
                     help='Option to set number of topic documents to output, default is 5')
args = parser.parse_args()

if args.number_topics == None:  
    n_topics = 20
else:
    n_topics = int(args.number_topics[0])

if args.number_words == None:  
    n_words = 10
else:
    n_words = int(args.number_words[0])
    
if args.number_docs == None:  
    n_docs = 5
else:
    n_docs = int(args.number_docs[0])



# load data
print('\nLoading data...')
infile = args.input_file
df = pd.read_csv(infile)
print('Input file:', infile)
print('Number comments:', df.shape[0])



# clean and preprocess data
print('\nCleaning data...')


# function to remove duplicate and very short comments
def clean_comments(df): 
    docs = [comment for comment in df['comment'].unique()]
    docs = [comment for comment in docs if  len(comment.split()) > 10]
    print('Number comments after cleaning:', len(docs))
    return (docs)
docs = clean_comments(df)


# function to lemmatise words in comments
# ref https://www.nltk.org/_modules/nltk/stem/wordnet.html 
def lemmatiser(unlemmatised_data): 
    lemmatised_data = []
   
    for item in unlemmatised_data:
        lemmatised_item = ""  
        item = item.replace("-", " ")
        words = word_tokenize(item)
              
        for word in words: 
            if word != WordNetLemmatizer().lemmatize(word.lower()): # noun lemmatiser
                lemmastised_word = WordNetLemmatizer().lemmatize(word.lower())
            else: # verb lemmatiser
                lemmastised_word = WordNetLemmatizer().lemmatize(word.lower(), "v") 
            lemmatised_item = lemmatised_item + lemmastised_word + " "
        lemmatised_data.append(lemmatised_item)
        
    return lemmatised_data  
lemmatised_docs = lemmatiser(docs)


                                    
# do lda                                  
print('\nGenerating topics...')
    
# https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
vectorizer = CountVectorizer(max_df = 0.5,
                             min_df = 5, 
                             max_features = None,
                             token_pattern = r'\b[a-zA-Z]{3,}\b',
                             stop_words = 'english')
doc_tf_vecs = vectorizer.fit_transform(lemmatised_docs)  # doc-tf matrix
doc_tf_words = vectorizer.get_feature_names()
    
model = LatentDirichletAllocation(n_components = n_topics, 
                                  learning_method = 'online',
                                  random_state=0)   
model.fit(doc_tf_vecs)
topic_word_distribs = model.components_
doc_topic_distribs = model.transform(doc_tf_vecs)
    


# represent topics
# extract dictionary of top n words from topics
def topic_words(topic_word_distribs, doc_tf_words, n_words):
    top_words_dic = {}
    
    for topic_idx, topic_probs in enumerate(topic_word_distribs):
        top_words = [doc_tf_words[i] for i in topic_probs.argsort()[:-n_words-1:-1]]
        top_words_dic[topic_idx] = top_words

    return top_words_dic
topic_wordsn = topic_words(topic_word_distribs, doc_tf_words, n_words)
    

# extract dictionary of top n doc indexes from topics
def top_ndoc_idxs(n_topics, doc_topic_distribs, n_docs):
    top_docs_dic = {}
    
    for topic_idx in range(n_topics):
        
        doc_probs = doc_topic_distribs[:,topic_idx]
        sorted_idxs = np.argsort(doc_probs)[::-1]  
        top_docs = sorted_idxs[0:n_docs]  
        top_docs_dic[topic_idx] = top_docs
        topic_idx += 1
    
    return top_docs_dic
top_doc_idxs = top_ndoc_idxs(n_topics, doc_topic_distribs, n_docs)
    
      
# for each topic, print topic words and comments to file
print('Topics saved to:', infile[:-4]+"_topics.txt")
outfile = open(infile[:-4]+"_topics.txt", "w+")
for topic_idx in range(n_topics):
    outfile.write("\n\n"+str(topic_wordsn[topic_idx]))
    for doc_idx in top_doc_idxs[topic_idx]:
        outfile.write('\n--'+str(docs[doc_idx].translate({ ord(c):' ' for c in '\n\t'})))   
outfile.close()

    
                                  
                                  
                            
