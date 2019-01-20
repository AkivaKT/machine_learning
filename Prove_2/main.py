# -*- coding: utf-8 -*-


from sklearn import datasets  # dataset
from sklearn.neighbors import KNeighborsClassifier  # algorithms
from sklearn.naive_bayes import GaussianNB
from knnclassifier import KnnClassifier
from randomclassifier import RandomClassifier
from sklearn.model_selection import train_test_split  # the split function

import pandas as pd    # other libraries
import numpy as np

    
def manu():
    while True:
        al = select_algorithm()
        if al == 0:
            break
        
        data, target = select_data()
        
        split = select_ratio()
        
        while True:
            seed = input('Enter a random number(seed): ')
            try:
                int(seed)
                break
            except:
                print('Enter while number.')
        
        ex_train, ex_test, de_train, de_test = train_test_split(data, target, test_size = split, random_state = int(seed))
    
        de_train = np.ravel(de_train)
        de_test = np.ravel(de_test)
        
        al.fit(ex_train, de_train)
        ALscore = al.score(ex_test, de_test)
        print('\naccuracy : {}'.format(ALscore))
        
        run = input('Do you wish to quit: (y/n): ')
        if run.upper() == 'Y':
            break
        
def select_algorithm():
    while True:
        print('Select classification algorithm:\n\t1. GaussianNB\n\t2. KNeighborsClassifier', 
          '\n\t3. RandomClassifier\n\t4. Hard-coded KnnClassifier \n\t0. Enter 0 to quit')
        al = int(input(''))
        if al == 1:
            classifier = GaussianNB()
        elif al == 2:
            k = input('Enter the value of K(nearest neighbor(s)):')
            classifier = KNeighborsClassifier(int(k))
        elif al == 3:
            classifier = RandomClassifier()
        elif al == 4:
            k = input('Enter the value of K(nearest neighbor(s)):')
            classifier = KnnClassifier(int(k))
        if al <= 4 and al >= 1:
            return classifier
        elif al != 0:
            print('invaild input, try again.')
        else:return 0
        
        
        
def select_data():
    while True:
        print('Select dataset:\n\t1. Iris\n\t2. Civic vs Corolla')
        dat = int(input(''))
        
        if dat == 1:
            data = datasets.load_iris()
            d = pd.DataFrame(data.data)
            t = pd.DataFrame(data.target)
        elif dat == 2:
            data = pd.read_csv('./CivicCorolla.csv')
            d = data[['Mileage','Price','Year']]
            t = data['Model']
        if dat <=4 and dat >=1:
            return d,t
        elif dat != 0:
            print('invaild input, try again.')
        else:return 0, 0
        


def select_ratio():
    while True:
        print('How much data do you want to use for the training model?\nEnter whole number(1-99)')
        train_ratio = float(input(''))/100
        if train_ratio < 1:
            return train_ratio
        else:print('invaild input, try again.')
            

if __name__ == "__main__":
    manu()
    
    
    



