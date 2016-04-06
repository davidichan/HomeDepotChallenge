#This script takes works towards calculating the cosine similarity between the tfidf document term matrices
#created from the search terms and the descriptions terms. The steps are as follows
#1. Read in tiny subset of training set and descriptions set (for speed and assessing terms)
#2. Stem, lemmatize, and process stop words for descriptions to create dtms based on term frequency and based on tfidf 
#3. As step 2, but for search terms and product title terms
#4. Take the search matrix from step 3 and apply to the dictionary of words found in the descriptions matrix.
#5. Calculate the cosine similarity between the each search term and the description of the matching product. 
##To run, ensure the libraries are installed and that the training and description sets are downloaded. 

library(tm)
library(SnowballC)
library(lsa)
#SnowballC for stemming

descriptions_full <- read.csv("product_descriptions.csv", header=T)

descriptions_sample <- descriptions_full[1:100,]
descriptions_sample$product_description <- as.character(descriptions_sample$product_description)
descriptions_sample$product_description <- removePunctuation(descriptions_sample$product_description)

#NB: need to pay attention to data type that each feature is; by default, descriptions are treated as levels, 
#which tm does not treat as text. Therefore, convert to character type first:
descriptions_sample[] <- lapply(descriptions_sample, as.character) 

#To add the product ID to the metadata, follow instructions given here:
#http://stackoverflow.com/questions/24501514/keep-document-id-with-r-corpus

myReader <- readTabular(mapping=list(content="product_description", id="product_uid"))
descrip_corpus <- VCorpus(DataframeSource(descriptions_sample), readerControl = list(reader=myReader))
#The result of this command is a corpus containing all the data from the dataframe - that is, 
#both the descriptions and the product_uid. 

#Apply transformations to process text: 
descrip_corpus <- tm_map(descrip_corpus, stripWhitespace)
descrip_corpus <- tm_map(descrip_corpus, content_transformer(tolower))
descrip_corpus <- tm_map(descrip_corpus, removeWords, stopwords("english"))
descrip_corpus <- tm_map(descrip_corpus, stemDocument, lazy = T)

#Stem completion:
#We complete the stems with the objective to reverse the stemming process so that the text 
#looks more 'normal' and readable. By passing the tm_map() the stemCompletion() function, 
#we can complete the stems of the documents in tweets.corpus, with the tweets.corpus.copy 
#as the dictionary. The default option is set to complete the match with the highest frequency term.

#The following line gives you the Document term matrix - i.e. a matrix of documents and words and
#how many times each word appears. 
dtm_descriptions <- DocumentTermMatrix(descrip_corpus)

#This function can also be used to create the tf-idf matrix, if the weighting is specified to use TFIDF, as follows
#note the normalize term will normalize the TF term - i.e. take the term frequency matrix and normalize based on the max word count in the document
tfidf_descriptions <-DocumentTermMatrix(descrip_corpus,control = list(weighting = function(x) weightTfIdf(x, normalize = TRUE)))



################################################
################################################
#Do the same thing, but for the training set:
################################################
################################################
train_sample <- read.csv("train.csv", header=T)

#NB: need to pay attention to data type that each feature is; by default, descriptions are treated as levels, 
#which tm does not treat as text. Therefore, convert to character type first:
train_sample[] <- lapply(train_sample, as.character) 
train_sample$search_term <- removePunctuation(train_sample$search_term)
train_sample$product_title <- removePunctuation(train_sample$product_title)

#To add the product ID to the metadata, follow instructions given here:
#http://stackoverflow.com/questions/24501514/keep-document-id-with-r-corpus

myReader_train_search <- readTabular(mapping=list(content="search_term", id="product_uid", sampleid="id", relevance="relevance"))
myReader_train_title <- readTabular(mapping=list(content="product_title", id="product_uid", sampleid="id", relevance="relevance"))
train_corpus_search <- VCorpus(DataframeSource(train_sample), readerControl = list(reader=myReader_train_search))
train_corpus_title <- VCorpus(DataframeSource(train_sample), readerControl = list(reader=myReader_train_title))
#The result of this command is a corpus containing all the data from the dataframe - that is, 
#both the descriptions and the product_uid. 

#Apply transformations to process text:
#On search terms:  
train_corpus_search <- tm_map(train_corpus_search, stripWhitespace)
train_corpus_search <- tm_map(train_corpus_search, content_transformer(tolower))
train_corpus_search <- tm_map(train_corpus_search, removeWords, stopwords("english"))
train_corpus_search <- tm_map(train_corpus_search, stemDocument, lazy = T)

#On product titles: 
train_corpus_title <- tm_map(train_corpus_title, stripWhitespace)
train_corpus_title <- tm_map(train_corpus_title, content_transformer(tolower))
train_corpus_title <- tm_map(train_corpus_title, removeWords, stopwords("english"))
train_corpus_title <- tm_map(train_corpus_title, stemDocument, lazy = T)

#Stem completion:
#We complete the stems with the objective to reverse the stemming process so that the text 
#looks more 'normal' and readable. By passing the tm_map() the stemCompletion() function, 
#we can complete the stems of the documents in tweets.corpus, with the tweets.corpus.copy 
#as the dictionary. The default option is set to complete the match with the highest frequency term.

#The following line gives you the Document term matrix - i.e. a matrix of documents and words and
#how many times each word appears. 
dtm_train_search <- DocumentTermMatrix(train_corpus_search)
dtm_train_title <- DocumentTermMatrix(train_corpus_title)

#This function can also be used to create the tf-idf matrix, if the weighting is specified to use TFIDF, as follows
#note the normalize term will normalize the TF term - i.e. take the term frequency matrix and normalize based on the max word count in the document
tfidf_train_search <-DocumentTermMatrix(train_corpus_search,control = list(weighting = function(x) weightTfIdf(x, normalize = TRUE)))
tfidf_train_title <-DocumentTermMatrix(train_corpus_title,control = list(weighting = function(x) weightTfIdf(x, normalize = TRUE)))

##############################
#Comparing the matrices:
###############################
#Converting matrices containing term frequency values
dtm_train_title.mat <- as.matrix(dtm_train_title)
dtm_train_search.mat <- as.matrix(dtm_train_search)
dtm_descriptions.mat <- as.matrix(dtm_descriptions)

#Converting matrices containing tfidf values
tfidf_train_title.mat <- as.matrix(tfidf_train_title)
tfidf_train_search.mat <- as.matrix(tfidf_train_search)
tfidf_descriptions.mat <- as.matrix(tfidf_descriptions)

#Keep the columns that intersect between the search dtm and the descriptions dtm
#xx <- data.frame(dtm_train_search.mat[,intersect(colnames(dtm_train_search.mat),colnames(dtm_descriptions.mat))])
#yy <- read.table(textConnection(""), col.names = colnames(dtm_descriptions.mat),colClasses = "integer")

#library(plyr)
#zz <- rbind.fill(xx, yy)


##Does same thing as last 4 lines of code - except better
#This takes the corpus containing the stemmed training set and converts it to a dtm using the dtm terms of the descriptions
#dtmtfidf_train_search_standardized <- DocumentTermMatrix(train_corpus_title_stemmed, control = list (weighting = function(x) weightTfIdf(x, normalize = TRUE), dictionary=Terms(dtm_descriptions)))
dtm_train_search_standardized <- DocumentTermMatrix(train_corpus_search_stemmed, control = list (dictionary=Terms(dtm_descriptions)))
dtm_train_search_standardized.mat <- as.matrix(dtm_train_search_standardized)

#Create a dtm containing the columns from the search dtm, mapping the description terms to the search term dtm
#This will result in a much simpler set of dtms, since there will be far fewer columns
dtm_descr_standardized <- DocumentTermMatrix(descrip_corpus_stemmed, control = list (dictionary=Terms(dtm_train_search)))
dtm_descr_standardized.mat <- as.matrix(dtm_descr_standardized)

tfidf_descr_standardized <- DocumentTermMatrix(descrip_corpus_stemmed, control = list (weighting = function(x) weightTfIdf(x, normalize = TRUE), dictionary=Terms(dtm_train_search)))
tfidf_descr_standardized.mat <- as.matrix(tfidf_descr_standardized)

#########################################
#########################################
#########################################
#At this point, the training set and descriptions are available as dtms both with absolute and tfidf values. Next steps:
#1. Match the docs in the training set to the columns of the descriptions - this will allow us to use blah_desc[descr_positions,] to effectively create a matrix containing 
#the descriptions that matches the training set line for line
#2. Apply the cosine function to each line in the training dtm to the descriptions dtm. > sapply(1:4, function(i) cosine(m1[i,], m2[i,]))

productuid_training_descriptions.ndx <- match(train_sample$product_uid,descriptions_sample$product_uid)
dtm_descr_match_train <- dtm_descriptions.mat[productuid_training_descriptions.ndx,]
cosine_sim_descr_columns <- sapply(1:nrow(dtm_train_search_standardized), function(i) cosine(dtm_train_search_standardized.mat[i,],dtm_descr_match_train[i,]))
write.csv(cosine_sim_descr_columns, file="cosine_sim_descr_col_search_vs_descr.csv", row.names=F)

#Using the descriptions matrix containing the search term columns only: 
dtm_descr_match_train_reverse <- dtm_descr_standardized.mat[productuid_training_descriptions.ndx,]
cosine_sim_search_columns <- sapply(1:nrow(dtm_train_search_standardized), function(i) cosine(dtm_train_search.mat[i,],dtm_descr_match_train_reverse[i,]))
write.csv(cosine_sim_search_columns, file="cosine_sim_search_col_search_vs_descr.csv", row.names=F)

tfidf_descriptions_match_search_columns <- tfidf_descr_standardized.mat[productuid_training_descriptions.ndx,]
cosine_sim_tfidf_search_columns <- sapply(1:nrow(dtm_train_search_standardized), function(i) cosine(tfidf_train_search.mat[i,],tfidf_descriptions_match_search_columns[i,]))
write.csv(cosine_sim_tfidf_search_columns, file="cosine_sim_tfidf_search_col_search_vs_descr.csv", row.names=F)

