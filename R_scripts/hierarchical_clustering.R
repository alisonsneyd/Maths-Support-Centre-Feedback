"
Created on Feb 1 2019

@author: Gizem Intepe

This file takes an input.csv with comments to be analysed in a column called 
'Query' and outputs a dendrogram of all words. 

"



#-----------------------------------------

#DENDROGRAM

#-----------------------------------------

#generate tf-idf weighted term document matrix
corpus = Corpus(VectorSource(input$Query))
query.tdm = TermDocumentMatrix(corpus)
query.w.tdm = as.matrix(weightTfIdf(query.tdm))
empties = which(colSums(query.w.tdm) == 0)   #remove empty docs
query.matrix = query.w.tdm[, -empties]       #final tdm without empty docs
dim(query.matrix)

frequent.words = which(apply(query.matrix > 0 , 1, sum) > 20) #take words appeared more than 20 documents
term.matrix = t(query.matrix[frequent.words, ])

norm.term.matrix = term.matrix %*% diag(1 / sqrt(colSums(term.matrix ^ 2)))
# preserve column names (terms associated to each column)
colnames(norm.term.matrix) = colnames(term.matrix)
D = dist(t(norm.term.matrix), method = "euclidean") ^ 2 / 2  #compute cosine distances between each word.

#dendrogram
h = hclust(D, method = "ward.D2") #aggloramative clustering with Ward method.

library(dendextend)
library("ggplot2")
library("ggdendro")

d = as.dendrogram(h)
d = d %>% color_branches(k = 7, col = rainbow) %>% color_labels(k = 7, col =
                                                                  rainbow)
d = set(d, "labels_cex", 1.4)
d = set(d, "branches_lwd", 3.4)
par(mar = c(0, 7, 0, 5))
plot_horiz.dendrogram(d, side = TRUE)


#-----------------------------------