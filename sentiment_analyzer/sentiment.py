"""
    Author: Ankit Dutta <cruxbeaker>
    Naive Bayes spam detection for NLP
    dataset: http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
"""

from __future__ import print_function, division
from future.utils import iteritems
from builtins import range

import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

wordnet_lemmatizer = WordNetLemmatizer()

# from http://www.lextek.com/manuals/onix/stopwords1.html
stopwords = set(w.rstrip() for w in open('../stopwords.txt'))
# note: an alternative source of stopwords
# from nltk.corpus import stopwords
# stopwords.words('english')

# load the reviews
# data courtesy of http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
positive_reviews = BeautifulSoup(open('data/positive.review').read(), "lxml")
positive_reviews = positive_reviews.find_all('review_text')

negative_reviews = BeautifulSoup(open('data/negative.review').read(), "lxml")
negative_reviews = negative_reviews.find_all('review_text')

# there are more positive reviews than negative reviews
# so let's take a random sample so we have balanced classes
np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]

# we can also oversample the negative reviews
# diff = len(positive_reviews) - len(negative_reviews)
# idxs = np.random.choice(len(negative_reviews), size=diff)
# extra = [negative_reviews[i] for i in idxs]
# print(extra)
# negative_reviews += extra

def myTokenizer(s):
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    return tokens


# create a word-to-index map so that we can create our word-frequency vectors later
# let's also save the tokenized versions so we don't have to tokenize again later
word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []

for review in positive_reviews:
    tokens = myTokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

for review in negative_reviews:
    tokens = myTokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1


# now let's create our input matrices
def tokensToVector(tokens, label):
    x = np.zeros(len(word_index_map) + 1) # last element is for the label
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum() # normalize it before setting label
    x[-1] = label
    return x

N = len(positive_tokenized) + len(negative_tokenized)
# (N x D+1 matrix - keeping them together for now so we can shuffle more easily later
data = np.zeros((N, len(word_index_map) + 1))
i = 0
for tokens in positive_tokenized:
    xy = tokensToVector(tokens, 1)
    data[i,:] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokensToVector(tokens, 0)
    data[i,:] = xy
    i += 1

# shuffle the data and create train/test splits
# try it multiple times!
np.random.shuffle(data)

X = data[:,:-1]
Y = data[:,-1]

# last 100 rows will be test
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print("Classification rate:", model.score(Xtest, Ytest))

# let's look at the weights for each word
# try it with different threshold values!
threshold = 0.5
for word, index in iteritems(word_index_map):
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print(word, weight)