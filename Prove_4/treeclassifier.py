# -*- coding: utf-8 -*-
import math as mt
from decision_node import DecisionNode



class Treeclassifier:
    def __init__(self):
        self.root = DecisionNode()
    
    def fit(self, ex_train, de_train):
        self.root.Isroot = True
        if len(ex_train) == len(de_train):
            self.root.build(ex_train, de_train)
        else:
            print(len(ex_train), len(de_train))
            print('data doesn\'t fit')

    def predict(self, ex_test):
        self.prediction = self.root.evaluate(ex_test)
        self.prediction = self.prediction.reindex(ex_test.index)
        
    def score(self, ex_test, de_test):
        self.predict(ex_test)
        self.correct = self.prediction[de_test.iloc[:,0] == self.prediction.iloc[:,0]]
        return(len(self.correct)/len(self.prediction))
        
    def get_path(self):
        self.root.display()
        
    def trim(self):
        self.root.combine_leaves()
        