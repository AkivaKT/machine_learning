# -*- coding: utf-8 -*-

class LeafNode:
    def __init__(self, t):
        self.isleaf = True
        self.target_class = t
        
    def __string__(self):
        return self.target_class
    
    def display(self, index):
        end_index = index - 1
        for i in range(1,end_index):
            print('\t', end = '')
        print('Prediction:{}, {} layers'.format(self.target_class,
              end_index))
        print('')
        
    def evaluate(self, ex_test):
        ex_test.insert(0, 'prediction', self.target_class)
        ex_test = ex_test.ix[:,['prediction']].copy(deep = True) 
        return ex_test
        
    def combine_leaves(self):
        self.target = self.target_class
        return [self.target_class]
        