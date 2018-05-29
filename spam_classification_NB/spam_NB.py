"""
    Author: Ankit Dutta <cruxbeaker>
    Naive Bayes spam detection for NLP
    dataset: https://archive.ics.uci.edu/ml/datasets/Spambase
"""

from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('spambase.data').as_matrix()     # use pandas for convenience
np.random.shuffle(data)     # shuffle each row in-place, but preserve the row

X = data[:, :48]
Y = data[:, -1]

# Split data into train and test set
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

# MultinomialNB Model
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
# Predict
print("Classification rate for NB: ", model.score(Xtest, Ytest))

# AdaBoost Model
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print("Classification rate for AdaBoost: ", model.score(Xtest, Ytest))