import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

UNKNOWN = 10000
data = pd.read_csv("training.txt", sep=" ", names = ["token","tag","chunking_tag"])
data_x = data["token"]
data_y = data["tag"]
tags = {}
unique_tags = data_y.unique()
def getTags(x,dict):
    if x in dict.keys():
        return dict[x]
    else:
        return UNKNOWN

for i in range(len(unique_tags)):
    tags[unique_tags[i]] = i
test = pd.read_csv("pos1.txt", sep=" ", names=["token", "tag", "chunking_tag"], engine='python')

vform = TfidfVectorizer()
x = vform.fit_transform()
test_x = test["token"]
test_y = test["tag"]
test_x = vform.transform(test_x)
y_encode = data_y.apply(lambda x: getTags(x,tags))
# Test data is 20%
trainX, testX, trainY, testY = train_test_split(x, y_encode, test_size=0.2, random_state=1, shuffle=True)


gaussian = GaussianNB()
gaussian.fit(trainX, trainY)
gaussian.predict(testX)
# Gaussian prediction ~ 44.3562459348932%

mnb = MultinomialNB()
mnb.fit(trainX, trainY)
mnb.predict(testX)
# MNB prediction ~ 70.0341380673431%

