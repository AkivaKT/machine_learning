# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 12:48:06 2019

@author: keith
"""
# Step 1
from sklearn import datasets
from sklearn.model_selection import train_test_split
from randomclassifier import RandomClassifier
import pandas as pd
import numpy as np
iris = datasets.load_iris()

iris_data = pd.DataFrame(iris.data)
iris_target = pd.DataFrame(iris.target)


# Step 2
## in as in independent variable, de as in dependent variable
in_train, in_test, de_train, de_test = train_test_split(iris_data, iris_target, test_size = .3, random_state = 98)

de_train = np.ravel(de_train)
de_test = np.ravel(de_test)

# Step 3
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(in_train, de_train)

# Step 4
ALscore = classifier.score(in_test, de_test)
print('accuracy: {}'.format(ALscore))


# Step 5  (check the randomclassifier file for more information for this class)

classifier = RandomClassifier()
classifier.fit(in_train, de_train)
new_score = classifier.score(in_test, de_test)
print('accuracy: {}'.format(new_score))




#above and beyond


dat = pd.read_csv('./CivicCorolla.csv')

in_train, in_test, de_train, de_test = train_test_split(dat[['Mileage','Price','Year']], dat['Model'], test_size=.3, random_state = 28)

de_train = np.ravel(de_train)
de_test = np.ravel(de_test)

classifier = GaussianNB()
classifier = GaussianNB()
classifier.fit(in_train, de_train)

ALscore = classifier.score(in_test, de_test)
print('accuracy: {}'.format(ALscore))


classifier = RandomClassifier()
classifier.fit(in_train, de_train)
new_score = classifier.score(in_test, de_test)
print('accuracy: {}'.format(new_score))


