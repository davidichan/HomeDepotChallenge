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

#Select subset of data
descriptions_sample <- descriptions_full[1:100,]

#Convert data type to character, in order for removePunctuation to work
descriptions_sample$product_description <- as.character(descriptions_sample$product_description)
descriptions_sample$product_description <- removePunctuation(descriptions_sample$product_description)

#Likely redundant line: 
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

