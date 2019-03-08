# -*- coding: utf-8 -*-
from layer import Layer
import math as mt
import numpy as np
import pandas as pd

class Nnetwork:
    def __init__(self):
        self.layers = []
        self.record = []
        self.learning = 0.2
        self.built = False

    def build(self, layer, node, output):
        if layer == len(node):
            layer_index = 0
            for i in range(layer - 1):
                new_layer = Layer()
                new_layer.add_node(node[layer_index], node[layer_index + 1])
                new_layer.add_bias(node[layer_index + 1])
                layer_index += 1
                self.layers.append(new_layer)
                
            output_layer = Layer()
            output_layer.is_output = True
            output_layer.output_node(output)
            self.layers.append(output_layer)
            self.built = True
            
        else:
            print('Layers and node numbers don\'t match')
            
    
    def train(self, ex_train, in_train): 
        if self.built:
            self.record = []
            for i in range(len(ex_train)):
                inp = np.array(ex_train.iloc[i])
                tar = in_train[i]
                
                for i in range(len(self.layers)):
                    if not self.layers[i].is_output:
                        inp = self.activate(self.layers[i].forward(inp))
                        if not self.layers[ i + 1 ].is_output:
                            self.layers[i + 1].store(inp)
                        self.layers[i + 1].activation = inp
                    else:
                        prediction = self.tell(self.layers[i], inp, tar)
                        self.record.append(prediction)
                self.backward()
                
        else:
            print('Model is not built yet.')
            
    def activate(self, inp):
        compute = lambda x: 1/(1 + mt.pow(mt.e, -x))
        return [compute(i) for i in inp]
            
    
    def run(self, ex_test):
        prediction = []
        for i in range(len(ex_test)):
                inp = np.array(ex_test.iloc[i])            
                for i in range(len(self.layers)):
                    if not self.layers[i].is_output:
                        inp = self.activate(self.layers[i].forward(inp))
                        if not self.layers[ i + 1 ].is_output:
                            self.layers[i + 1].store(inp)
                        self.layers[i + 1].activation = inp
                    else:
                        tar_class = self.tell_without(self.layers[i], inp)
                        prediction.append(tar_class)                  
        return prediction

            
    def backward(self):
        for i in range (len(self.layers), 1, -1):
            errors = []
            old_weights = []
            for node in self.layers[i - 1].nodes:
                if not node.bias:
                    errors.append(node.error)
            for node in self.layers[i - 2].nodes:
                if not node.bias:
                    old_weights = node.weights
                    node_e = np.array(old_weights) @ errors
                    node.error = node.value*(1-node.value) * node_e
                    
                node.weights =  np.array(node.weights) - ([self.learning] * np.array(errors) * [node.value])


                    
    def tell(self, layer, inp, tar):
        index = 0
        target = 0
        for i in range(len(inp)):
            out_node = layer.nodes[i]
            out_value = inp[i]
            if out_value >.5:
                index += 1
            if out_node.target == tar:
                out_node.value = 1
                if out_value >.5:
                    target = 1
            else:
                out_node.value = 0
            out_node.error = self.get_error(out_value, out_node.value, True)
        return (index == 1 and target)
            
    def get_error(self, out_value, target, output_node):
        if output_node:
            return out_value*(1 - out_value)*(out_value - target)
            
        
    def tell_without(self, layer, inp):
        index = np.argmax(inp)
        return layer.nodes[index].target
        

