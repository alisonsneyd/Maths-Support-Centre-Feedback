# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 14:57:04 2018

@author: Alison

This file contains a function to make a lemmatised version of the documents
before vectorisation.

"""

# imports
#import nltk
#nltk.download()
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize



# make lemmastised version of docs, input is list of strings (docs)
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


