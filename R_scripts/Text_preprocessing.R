"
Created on Feb 1 2019

@author: Gizem Intepe

This file takes an input.csv with comments to be analysed in a column called 
'Query' and cleans the data in Query column. 

"
input=read.csv("input.csv",as.is=TRUE)

library("tm") 

corpus = Corpus(VectorSource(input$Query))  #convert data to corpus object

corpus = tm_map(corpus, function(x) iconv(x, to='ASCII', sub=' ')) # remove special characters
corpus = tm_map(corpus, removeNumbers) # remove numbers
corpus = tm_map(corpus, stripWhitespace) # remove whitespace
corpus = tm_map(corpus, tolower) # convert all to lowercase
corpus = tm_map(corpus, removePunctuation) # remove punctuation
myStopList=c(stopwords(),"workshop","question", "understand", "also","student","students","need","needed","help",
             "hour","library","roving","rove","visit","mesh","sfb","maths","stat","stats","statistics",
             "mathematics","etc","etc.", "assignment", "especially", "particularly","etc",
             "wanted","want","work","worked","working","suggest","suggested","get","question","questions","problem",
             "exam","lecture","example","use","involve","involving","include","including","class",
             "problems","mathematical","returned","several","visits","visit","papers","paper", 
             "used","using","use","various", "understand","understanding" , "week")
corpus = tm_map(corpus, removeWords, myStopList )
corpus = tm_map(corpus, stemDocument) # convert all words to their stems

query = data.frame(text=sapply(corpus, identity), stringsAsFactors=F) #save text as a separate data frame
input$Query=query$text  #replace Query column of the original file with the cleaned text


#-------------------------------------------------------------------------------------

