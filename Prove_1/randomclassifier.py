# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 10:26:20 2019

@author: keith
"""
import pandas as pd
import numpy as ny

class RandomClassifier():
     
    def __init__(self):
        self.in_train = []
        self.de_train = []
        self.prediction = []

    def fit(self, in_train, de_train):
        self.in_train = in_train
        self.de_train = de_train
  
    '''
    This classifier will find the unique values of the outcome and 
    randomly generate a prediction from the list of the unique vales
    '''      
    def predict(self, in_test):
        
        self.prediction = []
        
        for row in range(len(in_test)):
            random_output = ny.random.choice(self.de_train)
            self.prediction.append(random_output)

        return self.prediction
    
    def score(self, in_test, de_test):
        
        self.prediction = self.predict(in_test)
        correct = 0
        count = 0
        for i in range(len(de_test)):
            if self.prediction[i] == de_test[i]:
                correct += 1
            count += 1
            
        return(correct/count)