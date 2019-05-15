"
Created on Feb 1 2019

@author: Gizem Intepe

This file takes an input.csv with comments to be analysed in a column called 
'Query' and outputs unigram and bigram wordclouds. 

"


#--------------------
# WORDCLOUDS
#--------------------

library(wordcloud)
library(RColorBrewer)

#generating term document matrix (tdm)
corpus = Corpus(VectorSource(input$Query))
query.tdm = as.matrix(TermDocumentMatrix(corpus))
empties = which(colSums(query.tdm) == 0)
query.matrix = query.tdm[, -empties] #final tdm

freqs = rowSums(query.matrix)  #unigrams

pal2 <- brewer.pal(12, "Paired")
wordcloud(
  names(freqs),
  freqs,
  random.order = FALSE,
  min.freq = 30,
  scale = c(3, 1),
  colors = pal2[-11],
  random.color = TRUE
)


#generating bigram wordclouds
#----------------------------
library(tau)

bigrams = textcnt(input$Query, n = 2, method = "string")
bigrams = bigrams[order(bigrams, decreasing = TRUE)]  #bigrams

write.csv(bigrams, file = paste("bigrams.csv")) #make a copy for later use

wordcloud(
  names(bigrams) ,
  bigrams,
  random.order = FALSE,
  min.freq = 16,
  scale = c(3, 0.1),
  colors = pal2[-11],
  random.color = TRUE
)


#-------------------------------------------------------------------------------------

