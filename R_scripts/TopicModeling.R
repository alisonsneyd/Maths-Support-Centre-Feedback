"
Created on Feb 1 2019

@author: Gizem Intepe

This file takes an input.csv with comments to be analysed in a column called 
'Query' and implements LDA. The output is a .csv file of topics, represented 
by their most probable words.

"


#-----------------------------------------

#TOPIC MODELS

#-----------------------------------------

#generating Document Term Matrix (dtm)
corpus = Corpus(VectorSource(input$Query))
query.dtm = DocumentTermMatrix(corpus) # create a DocumentTermMatrix object

#remove empty documents in dtm
rowTotals = apply(query.dtm , 1, sum) #Find the sum of words in each Document
empty.ids = which(rowTotals == 0)
querydtm.new = query.dtm[rowTotals > 0,] #remove all docs without words
querydtm.new = as.matrix(querydtm.new)
dim(querydtm.new)  #check the dimension



# Analysis of the best number of topics
#--------------------------------------
 
library(ldatuning) # use this package to choose best number of topics

result <- FindTopicsNumber(
  input$Query,
  topics = seq(from = 2, to = 45, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 2L,
  verbose = TRUE
)

FindTopicsNumber_plot(result)
# best number of topics occurs at minimized Arun2010 and CaoJuan2009; maximized Griffiths2004 and Deveaud2014


#Latent Dirichlet Allocation (LDA)
#--------------------------------

library("topicmodels")

#Set parameters for Gibbs sampling
burnin = 4000
iter = 2000
thin = 500
seed = list(2003, 5, 63, 100001, 765)
nstart = 5
best = TRUE


#Number of topics
k = 13  #change this depending on your finding from best number of topics

#Run LDA using Gibbs sampling
ldaOut = LDA(
  input$Query,
  k,
  method = "Gibbs",
  control = list(
    nstart = nstart,
    seed = seed,
    best = best,
    burnin = burnin,
    iter = iter,
    thin = thin
  )
)

#write out results to a .csv file
ldaOut.topics <- as.matrix(topics(ldaOut))  #assign documents to topics
write.csv(ldaOut.topics, file = paste("LDAGibbs", k, "DocsToTopics.csv")) 

#top 6 terms in each topic,
#change to a different number if you want to find differrent number of words in each topic
ldaOut.terms <- as.matrix(terms(ldaOut, 6))    #top 6 terms in each topic
write.csv(ldaOut.terms, file = paste("level_0_topics", k, "TopicsToTerms.csv")) 

#probabilities associated with each topic assignment
topicProbabilities <- as.data.frame(ldaOut@gamma) #topic probabilities
write.csv(topicProbabilities,
          file = paste("LDAGibbs", k, "TopicProbabilities.csv"))  


#-------------------------------------------------------------------------------------

