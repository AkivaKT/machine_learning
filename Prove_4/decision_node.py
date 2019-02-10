# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from leaf_node import LeafNode

class DecisionNode:
    def __init__(self):
        self.Isroot = False
        self.child = {}
        # data
        self.ex = 0
        # target
        self.de = 0

    def build(self, ex_train, de_train):
        self.ex = ex_train
        self.de = de_train
        # End case 1 
        # If there is only one class in the target data, return leafnode
        class_array = de_train.iloc[:,0].unique()
        if len(class_array) == 1:
            if self.Isroot == False:
                return LeafNode(class_array[0])
            else: 
                self.node = ex_train.columns[0]
                self.child['end'] = LeafNode(class_array[0])

        # find the column that have to best information gain
        gain, column = self.compute_gain(ex_train, de_train)
        # End case 2
        # Check if there is any features worth parting the data left, if false
        # end with leaf node
        if column == 'none':    
            if self.Isroot == False:
                return LeafNode(max(de_train.iloc[:,0]))
            else: 
                self.child['end'] = LeafNode((max(de_train.iloc[:,0])))
                return

        self.node = column
        # if gain is greater than 0, we will part the data
        for value in ex_train[column].unique():
            parted_data = ex_train[ex_train[column] == value]
            # drop extra columns
            parted_data = self.drop_col(parted_data, column)
            parted_target = de_train[ex_train[column] == value]
            
            class_array = parted_target.iloc[:,0].unique()
            column_array = []
            for columns in parted_data:
                column_array.append(len(parted_data[columns].unique()))
            # End case 3
            # if a group has only one target class, return leafnode
            if len(class_array) == 1:
                self.child[value] = LeafNode(class_array[0])
            #End case 4
            # if a group's variables only hold a value across all columns
            # append a leaf node
            elif sum(column_array) == parted_data.shape[1]:
                self.child[value] = LeafNode(max(parted_target.iloc[:,0]))
            else:
                # if all end cases are false,  create a DecisionNode and 
                # and build brenches.
                self.child[value] = DecisionNode()
                self.child[value].build(parted_data, parted_target)
        
    def compute_gain(self, ex_train, de_train):
        best_gain = 100000
        best_column = 'none'
        # loop through each column in the train data
        for col in ex_train:
            entropy = []
            if len(ex_train[col].unique()) == 1:
                continue
            # loop through each unique value in that column
            for value in ex_train[col].unique():
                group = de_train[ex_train[col] == value]
                # get a list of entropy of that value
                e = self.get_entropy(group.iloc[:,0].value_counts()/
                                                len(group))
                entropy.append(e)
            #look at the mean of the list of entropies and store the best one
            # the smellest one (best information gain)
            if np.mean(entropy) <= best_gain:
                best_gain = np.mean(entropy)
                best_column = col
        return best_gain, best_column

    def get_entropy(self, ratio):
        return (ratio*np.log2(ratio)).sum() * -1

    def drop_col(self, parted_data, column):
        parted_data = parted_data.drop(columns = column)
        data_back = parted_data.copy()
        # if column has the only one unique value, drop it
        for col in parted_data:
            if len(parted_data[col].unique()) == 1:
                data_back = data_back.drop(columns = col)
        # return columns left
        return data_back

    def display(self, index = 1):
        nodes = []
        for node in self.child:
            nodes.append(node)
        # print tab according to layer
        for i in range(1,index):
            print('\t', end = '')
        print(self.node, nodes)
        layer = index + 1
        for node in self.child:
            # print tab according to layer
            for i in range(1,index):
                print('\t', end = '')
            print('column: {}, value: "{}":'.format(self.node, node))
            # display layer recursively
            self.child[node].display(layer)
        return
        
    def __str__(self):
        return self.node
    
    def evaluate(self, ex_test):
        #create empty dataframe
        prediction = pd.DataFrame()
        # separate data according the column and values
        for value in ex_test[self.node].unique():
            ex_test_next = ex_test[ex_test[self.node] == value]
            # if value exsists in tree, predict using tree
            if value in self.child:
                frame = self.child[value].evaluate(ex_test_next)
                prediction = prediction.append(frame, sort = True)
            # if value doesn't exsist in tree, predict using probability 
            else:
                ex_test_next.insert(0, 'prediction', max(self.de.iloc[:,0]))
                naive = ex_test_next.ix[:,['prediction']].copy(deep = True) 
                prediction = prediction.append(naive, sort = True)
        return prediction
    
    def combine_leaves(self):
        target_list = []
        for child in self.child:
            targetlist = self.child[child].combine_leaves()
            if len(set(targetlist)) == 1 and len(targetlist) != 1:
                self.child[child] = LeafNode(targetlist[0])
                target_list.append(targetlist[0])
            else:
                for target in targetlist:
                    target_list.append(target)
        return target_list
            
    