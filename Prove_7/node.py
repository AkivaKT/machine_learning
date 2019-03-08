# -*- coding: utf-8 -*-

import random as rd
import numpy as np

class Node:
    def __init__(self):
        self.bias = False
        self.weights = []
        self.error = 0
        self.value = 0
        
    def add_weight(self, next_layer):
        for i in range (next_layer):
            weight = rd.uniform(-.5,.5)
            self.weights.append(weight)
        self.weights = self.weights
            
            
            
