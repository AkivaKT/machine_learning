# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from network import Nnetwork


class Nnclassifier:
    
    def __init__(self):
        self.data = pd.DataFrame()
        self.target = pd.DataFrame()
        self.network = Nnetwork()
        self.precent = []
        
    def setting(self, layer, node, max, learning):
        self.layer = layer
        self.node = node
        self.max = max
        self.network.learning = learning
        
    def valid(self, va_train, va_test):
        self.va_set = va_train
        self.va_target = va_test
        
    def fit(self, ex_train, in_train):
        self.data = ex_train
        self.target = in_train
        output_list = np.unique(in_train)
        self.node.append(len(output_list))
        self.network.build(self.layer, self.node, output_list)
        
        i = 0
        best = 0
        correct = 0
        count = 0
        
        while count < 250 and i < self.max and not self.stop(best, correct):
            i += 1
            self.network.train(ex_train, in_train)
            valid_set = self.network.run(self.va_set)
            correct = pd.Series(valid_set)[pd.Series(valid_set) == self.va_target].shape[0]
            self.precent.append(correct/len(self.va_target))
            
            if best > correct/len(self.va_target):
                count += 1
                
            if  correct/len(self.va_target) > best:
                best =  correct/len(self.va_target)
                
            print(i, 'Best:', best, "current:", correct/len(self.va_target), 
                  count)

    def stop(self, best, correct):
        
        
        if best == 1:
            return 1
        elif best > .95 and (best - correct/len(self.va_target) > 0.02):
            return 1
        elif best > .85 and (best - correct/len(self.va_target) > 0.025):
            return 1
        else:
            print('run')
            return 0
            
    
    def predict(self, ex_test, in_test):
        self.prediction = self.network.run(ex_test)
        return(self.network.run(ex_test))


