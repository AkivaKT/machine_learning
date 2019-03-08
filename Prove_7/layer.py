# -*- coding: utf-8 -*-
from node import Node
import numpy as np


class Layer:
    def __init__(self):
        self.is_output = False
        self.nodes = []
        self.col = 0
        self.n_node = 0
        
    def add_node(self, col, n_node):
        self.col = col
        self.n_code = n_node
        
        for i in range(col):
            self.nodes.append(Node())
        
        for node in self.nodes:
            node.add_weight(n_node)
            
    def output_node(self, output):
        for i in range(len(output)):
            op = Node()
            op.target = output[i]
            self.nodes.append(op)
        
        
    
    def add_bias(self, n_node):
        bias_node = Node()
        bias_node.bias = True
        bias_node.value = -1
        bias_node.add_weight(n_node)
        self.nodes.insert(0, bias_node)
        
    def store(self, inp):
        for i in range(len(inp)):
            self.nodes[i + 1].value = inp[i]
        
    def forward(self, observation):
        if len(observation) == len(self.nodes) - 1:   
            
            weight_array = np.empty((self.n_code, 1))
            for node in self.nodes[1:]:
                weight_array = np.c_[weight_array, node.weights]
            weight_array = np.delete(weight_array,0,1)
            node_values = np.array(weight_array @ observation)
            bias_value =  np.array(self.nodes[0].weights) * self.nodes[0].value
            x = np.add(node_values ,bias_value)
            return x
