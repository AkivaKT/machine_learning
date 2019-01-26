# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd 
from math import sqrt

'''
 This classifier to to minic the use of K nearest neighbor al
 '''
 
class KnnClassifier():
    def __init__(self, k):
        self.train = pd.DataFrame()
        self.k = k
        self.prediction = pd.DataFrame()
        
    def fit(self, ex_train, de_train):
        self.train = pd.DataFrame(ex_train)
        self.train_target = de_train
        
    def predict(self, ex_test):
        self.prediction = pd.DataFrame()
        d = lambda x,y:sqrt(np.sum((x-y) ** 2))
        for i, t_row in ex_test.iterrows():
            diff_and_class = []
            for j, s_row in self.train.iterrows():
                diff_and_class.append(d(t_row, s_row))
            diff_and_class = pd.DataFrame(diff_and_class)
            diff_and_class['Target'] = self.train_target
            knn = diff_and_class.nsmallest(self.k, 0)
            if len(knn['Target'].unique()) == self.k:
                result = int(knn.iloc[0]['Target'])
                self.prediction = self.prediction.append({'class': result}, 
                                                         ignore_index = True)
            else:
                result = knn['Target'].value_counts().index[0]
                self.prediction = self.prediction.append({'class': result}, 
                                                         ignore_index = True)
        self.prediction['class'] = self.prediction['class']
        
    def score(self,ex_test,de_test):
        self.predict(ex_test)
        correct = self.prediction['class'] == pd.DataFrame(de_test)[0]
        return (len(correct[correct == True]) / correct.shape[0])
            
            