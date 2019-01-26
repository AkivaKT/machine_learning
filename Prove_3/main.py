# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from knnclassifier import KnnClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


car_headers = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

cardat = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',
                   names = car_headers, index_col = False) 

auto_headers = ['mpg', 'cyl', 'displacement', 'hp', 'wt', 'acceleration', 
               'model_year', 'origin', 'car_name']

autodat = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original',
                     names = auto_headers, sep = '\s+')

student = pd.read_csv('student-mat.csv', sep = ';')


#cardat 1
print(cardat.isna().sum())

number_coded = {'vhigh' : 4, 'high'  : 3, 'med'   : 2, 'low'   : 1,
                'big'   : 3, 'small' : 1, '1'     : 1, '2'     : 2,
                '3'     : 3, '4'     : 4, '5more' : 5, 'more'  : 8,
                'unacc' : 1, 'acc'   : 2, 'good'  : 3, 'vgood': 4
                }

converting = {'buying'  : number_coded,
              'maint'   : number_coded,
              'safety'  : number_coded,
              'doors'   : number_coded,
              'persons' : number_coded,
              'lug_boot': number_coded,
              'class'   : number_coded
              }

cardat.replace(converting, inplace = True)

target = np.asanyarray(cardat['class'])
data   = cardat.iloc[:,0:5]

ex_train, ex_test, de_train, de_test = train_test_split(data, target,
                                      test_size = .3, 
                                      random_state = 98)

std_auto = preprocessing.StandardScaler().fit(ex_train)
ex_train_std = std_auto.transform(ex_train)
ex_test_std  = std_auto.transform(ex_test)

print('car dataset:')
for i in range(3,9):
    classifier = KNeighborsClassifier(i)
    classifier.fit(ex_train_std, de_train)
    al = classifier.score(ex_test_std, de_test)
    print('when k = {}:\nscore: {}'.format(i, al))


# dataset2
autodat.describe()


autodat.dropna(inplace = True)
autodat = autodat.iloc[:,0:7]
target  = np.asarray(autodat['mpg'])
data    = autodat.iloc[:,1:7]

ex_train, ex_test, de_train, de_test = train_test_split(data, target,
                                      test_size = .5, 
                                      random_state = 98)


std_auto = preprocessing.StandardScaler().fit(ex_train)
ex_train_std = std_auto.transform(ex_train)
ex_test_std  = std_auto.transform(ex_test)


print('auto dataset:')
for i in range(3,9):
    classifier = KNeighborsRegressor(i)
    classifier.fit(ex_train_std, de_train)
    al = classifier.score(ex_test_std, de_test)
    print('when k = {}:\nscore: {}'.format(i, al))
    
    
#student data
    
number_coded = {'GP' : 1, 'MS' : 0, 'F'   : 1, 'M'   : 0,
                'U'  : 1, 'R'  : 0, 'LE3' : 1, 'GT3' : 0,
                'T'  : 1, 'A'  : 0, 'yes' : 1, 'no'  : 0
               }

converting = {'school'    : number_coded,
              'sex'       : number_coded,
              'address'   : number_coded,
              'famsize'   : number_coded,
              'Pstatus'   : number_coded,
              'schoolsup' : number_coded,
              'famsup'    : number_coded,
              'paid'      : number_coded,
              'activities': number_coded,
              'nursery'   : number_coded,
              'higher'    : number_coded,
              'internet'  : number_coded,
              'romantic'  : number_coded
              }
    
student.replace(converting, inplace = True)


target  = np.asarray(student['G3'])
data    = student.iloc[:,0:32]

data_long = pd.get_dummies(data, 
               columns = ['Mjob', 'Fjob', 'reason', 'guardian'], dtype = 'int')

ex_train, ex_test, de_train, de_test = train_test_split(data_long, target,
                                      test_size = .3, 
                                      random_state = 488787,
                                      shuffle = True)

std_stu = preprocessing.StandardScaler().fit(ex_train)
ex_train_std = std_stu.transform(ex_train)
ex_test_std  = std_stu.transform(ex_test)

print('student dataset:')
for i in range(3,9):
    classifier = KNeighborsRegressor(i)
    classifier.fit(ex_train_std, de_train)
    al = classifier.score(ex_test_std, de_test)
    print('when k = {}:\nscore: {}'.format(i, al))












