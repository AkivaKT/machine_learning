# -*- coding: utf-8 -*-

import pandas as pd
from treeclassifier import Treeclassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

# dataset 1 (iris)
data = datasets.load_iris()
d = pd.DataFrame(data.data)
t = pd.DataFrame(data.target)
# convert numeric data to descrete data
for col in d:
    d[col] = d[col] < d[col].mean()

ex_train, ex_test, de_train, de_test = train_test_split(d, t, test_size = .25, 
                                                        random_state = 23)

print('iris dataset, sklearn tree classifier')
print('accuracy:')
t = DecisionTreeClassifier(criterion='entropy')
t.fit(ex_train,de_train)
print(t.score(ex_test, de_test))
print('iris dataset, my own tree classifier')
print('accuracy:')
r = Treeclassifier()
r.fit(ex_train, de_train)
print(r.score(ex_test, de_test))
print('path before trim:')
r.get_path()
r.trim()
print('path after trim:')
r.get_path()

#dataset 2
print('len dataset')
index_heading = ['instance', 'age', 'spectacle', 
                 'astigmatic', 'tear_production_rate',
                 'classes']
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/lenses/lenses.data',
                   names = index_heading, 
                   sep = '\s+')

col1 = {1 : 'Young',
        2 : 'Pre-presbyopic',
        3 : 'presbyopic'}

col2 = {1 : 'myope',
        2 : 'hypermetrope'}

col3 = {1 : 'No',
        2 : 'Yes'}

col4 = {1 : 'reduced',
        2 : 'normal'}

col5 = {1 : 'hard',
        2 : 'soft',
        3 : 'not fit'}

converting = {'age'       : col1,
              'spectacle' : col2,
              'astigmatic': col3,
              'tear_production_rate' : col4,
              'classes'   : col5}
data.replace(converting, inplace = True)

data = pd.get_dummies(data, 
               columns = ['age', 'spectacle', 
                          'astigmatic', 'tear_production_rate'],
                          dtype = 'int')

ta = pd.DataFrame(data.loc[:,'classes'])
d = data.drop(columns = ['instance', 'classes'])

ex_train, ex_test, de_train, de_test = train_test_split(d, ta, test_size = .4, 
                                                        random_state = 32356)

print('iris dataset, sklearn tree classifier')
print('accuracy:')
t = DecisionTreeClassifier(criterion='entropy')
t.fit(ex_train,de_train)
print(t.score(ex_test, de_test))
print('iris dataset, my own tree classifier')
print('accuracy:')
r = Treeclassifier()
r.fit(ex_train, de_train)
print(r.score(ex_test, de_test))
print('path before trim:')
r.get_path()
r.trim()
print('path after trim:')
r.get_path()

#dataset 3

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.data')

ta = pd.DataFrame(data.loc[:,'draw'])
d = data.drop(columns = ['draw'])

ex_train, ex_test, de_train, de_test = train_test_split(d, ta, test_size = .4, 
                                                        random_state = 32356)
print('chess dataset')
print('My own tree classifier:')
r = Treeclassifier()
r.fit(ex_train, de_train)
print('accuracy:')
print(r.score(ex_test, de_test))
r.trim()


print('Dataset , combining columns ')
data['White King'] = (data['a'] + data['1'].astype(str)).astype(str)
data['White Rook'] = (data['b'] + data['3'].astype(str)).astype(str)
data['Black King'] = (data['c'] + data['2'].astype(str)).astype(str)

ta = pd.DataFrame(data.loc[:,'draw'])
d = data.drop(columns = ['a', 'b', 'c', '1', '2', '3', 'draw'])

ex_train, ex_test, de_train, de_test = train_test_split(d, ta, test_size = .4, 
                                                        random_state = 32356)
print('My own tree classifier:')
r = Treeclassifier()
r.fit(ex_train, de_train)
print('accuracy:')
print(r.score(ex_test, de_test))
r.trim()

print('Dataset with one hot encoding ')
d = pd.get_dummies(d, 
               columns = ['White King', 'White Rook', 
                          'Black King'],
                          dtype = 'int')

ex_train, ex_test, de_train, de_test = train_test_split(d, ta, test_size = .4, 
                                                        random_state = 32356)

t = DecisionTreeClassifier(criterion='entropy')
t.fit(ex_train,de_train)
print('Sklearn classifier:')
print('accuracy:')
print(t.score(ex_test, de_test))