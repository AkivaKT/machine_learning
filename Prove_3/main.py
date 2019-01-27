# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from knnclassifier import KnnClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# load all the data sets
## car data
car_headers = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
cardat = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',
                   names = car_headers, index_col = False) 
cardat_1 = cardat.copy()

## auto data
auto_headers = ['mpg', 'cyl', 'displacement', 'hp', 'wt', 'acceleration', 
               'model_year', 'origin', 'car_name']
autodat = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original',
                     names = auto_headers, sep = '\s+')

## student data
student = pd.read_csv('student-mat.csv', sep = ';')

student_1 = student.copy()
student_2 = student.copy()
# cardat (treating non-numeric data by label encoding, and Find and place)
print('Cardata')
## checking for any missing data
print('missing data:\n', cardat.isna().sum(), '\n')
print('sample data before converting:\n', cardat.head(), '\n')

## data conversion
number_coded = {'vhigh' : 4, 'high'  : 3, 'med'   : 2, 'low'   : 1,
                'big'   : 3, 'small' : 1, '1'     : 1, '2'     : 2,
                '3'     : 3, '4'     : 4, '5more' : 5, 'more'  : 8,
                'unacc' : 1, 'acc'   : 2, 'good'  : 3, 'vgood' : 4
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

print('data conversion:')
for key in number_coded:
    print ('converting {} as {}'.format(key,number_coded[key]))

print('data after data conversion:\n', cardat.head())
# preping data for model
target = np.asanyarray(cardat['class'])
data   = cardat.iloc[:,0:6]
# slipting data
ex_train, ex_test, de_train, de_test = train_test_split(data, target,
                                      test_size = .3, 
                                      random_state = 98,
                                      shuffle = True)

print('data before standardisation:\n', ex_train.head())
# standardisation
std_auto = preprocessing.StandardScaler().fit(ex_train)
ex_train_std = std_auto.transform(ex_train)
ex_test_std  = std_auto.transform(ex_test)

print('data after standardisation(z-score):\n', pd.DataFrame(ex_train_std[1:5]))

# KNN classifier
print('\nresult:')
bestScore = 0
for i in range(3,9):
    classifier = KNeighborsClassifier(i)
    classifier.fit(ex_train_std, de_train)
    al = classifier.score(ex_test_std, de_test)
    print('when k = {}:\nscore: {}'.format(i, al))
    if al > bestScore: 
        bestScore = al
        bestK = i
print('Best perfornance:\nK:{}\nscore: {}'.format(bestK, bestScore))    


# let's use one hot encoding for all non-numeric column
print('\nAppling one hot encoding to car dataset and retest:')

# find and place the number string
converting = {'doors'   : number_coded,
              'persons' : number_coded
              }

cardat_1.replace(converting, inplace = True)


# preping data for model
target = np.asanyarray(cardat_1['class'])
data   = cardat_1.iloc[:,0:6]

# apply one hot encoding
data_long = pd.get_dummies(data, 
               columns = ['buying', 'maint', 'lug_boot', 'safety'], dtype = 'int')
print('data after one hot encoding:', data_long.head())
# slipting data
ex_train, ex_test, de_train, de_test = train_test_split(data_long, target,
                                      test_size = .3, 
                                      random_state = 988977,
                                      shuffle = True)

# standardisation
std_auto = preprocessing.StandardScaler().fit(ex_train)
ex_train_std = std_auto.transform(ex_train)
ex_test_std  = std_auto.transform(ex_test)

print('data after standardisation(z-score):\n', pd.DataFrame(ex_train_std[1:5]))

# KNN classifier
print('\nresult:')
bestScore = 0
for i in range(3,9):
    classifier = KNeighborsClassifier(i)
    classifier.fit(ex_train_std, de_train)
    al = classifier.score(ex_test_std, de_test)
    print('when k = {}:\nscore: {}'.format(i, al))
    if al > bestScore: 
        bestScore = al
        bestK = i
print('Best perfornance:\nK:{}\nscore: {}'.format(bestK, bestScore))    


# dataset2
print('Auto data set\nMissing data:')
print('data set has {} rows.'.format(autodat.shape[0]))
print(autodat.isna().sum())

#dropping nan(s):
autodat.dropna(inplace = True)
print('\nmissing data removed')
print('data set has {} rows.'.format(autodat.shape[0]))
autodat = autodat.iloc[:,0:7]
target  = np.asarray(autodat['mpg'])
data    = autodat.iloc[:,1:7]

# print data
print(autodat.head())
print('all columns are numeric, no conversion needed')
ex_train, ex_test, de_train, de_test = train_test_split(data, target,
                                      test_size = .5, 
                                      random_state = 8,
                                      shuffle = True)

# standardisation
std_auto = preprocessing.StandardScaler().fit(ex_train)
ex_train_std = std_auto.transform(ex_train)
ex_test_std  = std_auto.transform(ex_test)

print('data after standardisation(z-score):')
print(pd.DataFrame(ex_train_std).head())

print('\nresult:')
bestScore = 0
for i in range(3,9):
    classifier = KNeighborsRegressor(i)
    classifier.fit(ex_train_std, de_train)
    al = classifier.score(ex_test_std, de_test)
    print('when k = {}:\nscore: {}'.format(i, al))
    if al > bestScore: 
        bestScore = al
        bestK = i
 
print('Best perfornance:\nK:{}\nscore: {}'.format(bestK, bestScore))    



#student data
print('student data')
print('missing data:')
print(student.isna().sum())
print('\ntest 1 on student data (one hot encoding)')

print('sample data before one hot encoding:', student.head())

# preping data for model
target = np.asanyarray(student['G3'])
data   = student.iloc[:,0:32]

# apply one hot encoding
data_long = pd.get_dummies(data, 
               columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 
                          'schoolsup', 'famsup' , 'paid', 'activities','nursery',
                          'higher', 'internet', 'romantic', 'Mjob', 'Fjob', 
                          'reason', 'guardian'], dtype = 'int')


print('sample data after one hot encoding:', data_long.head())
print('there is {} columns after one hot encoding.'.format(data_long.shape[1]))

ex_train, ex_test, de_train, de_test = train_test_split(data_long, target,
                                      test_size = .3, 
                                      random_state = 488787,
                                      shuffle = True)

std_stu = preprocessing.StandardScaler().fit(ex_train)
ex_train_std = std_stu.transform(ex_train)
ex_test_std  = std_stu.transform(ex_test)

print('\nresult:')
bestScore = 0
for i in range(3,9):
    classifier = KNeighborsRegressor(i)
    classifier.fit(ex_train_std, de_train)
    al = classifier.score(ex_test_std, de_test)
    print('when k = {}:\nscore: {}'.format(i, al))
    if al > bestScore: 
        bestScore = al
        bestK = i
    

print('Best perfornance:\nK:{}\nscore: {}'.format(bestK, bestScore))    




print('\ntest 2 on student data (label encoding)')

number_coded = {'GP'       : 1, 'MS'       : 0, 'F'       : 1, 'M'       : 0,
                'U'        : 1, 'R'        : 0, 'LE3'     : 1, 'GT3'     : 0,
                'T'        : 1, 'A'        : 0, 'yes'     : 1, 'no'      : 0,
               'other'     : 0, 'services' : 1, 'at_home' : 2, 'teacher' : 3,
               'health'    : 4, 'health'   : 5, 'course'  : 1, 'home'    : 2,
               'reputation': 3, 'mother'   : 1, 'father'  : 2}


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
              'romantic'  : number_coded,
              'Mjob'      : number_coded,
              'Fjob'      : number_coded,
              'reason'    : number_coded,
              'guardian'  : number_coded
              }
student.replace(converting, inplace = True)


target  = np.asarray(student['G3'])
data    = student.iloc[:,0:32]

ex_train, ex_test, de_train, de_test = train_test_split(data_long, target,
                                      test_size = .3, 
                                      random_state = 488787,
                                      shuffle = True)

std_stu = preprocessing.StandardScaler().fit(ex_train)
ex_train_std = std_stu.transform(ex_train)
ex_test_std  = std_stu.transform(ex_test)

print('\nresult:')
bestScore = 0
for i in range(3,9):
    classifier = KNeighborsRegressor(i)
    classifier.fit(ex_train_std, de_train)
    al = classifier.score(ex_test_std, de_test)
    print('when k = {}:\nscore: {}'.format(i, al))
    if al > bestScore: 
        bestScore = al
        bestK = i
    

print('Best perfornance:\nK:{}\nscore: {}'.format(bestK, bestScore))    



print('\ntest 3 on student data (both label encoding and one hot encoding(custom))')



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
    
student_1.replace(converting, inplace = True)


target  = np.asarray(student_1['G3'])
data    = student_1.iloc[:,0:32]

data_long = pd.get_dummies(data, 
               columns = ['Mjob', 'Fjob', 'reason', 'guardian'], dtype = 'int')

ex_train, ex_test, de_train, de_test = train_test_split(data_long, target,
                                      test_size = .3, 
                                      random_state = 488787,
                                      shuffle = True)

std_stu = preprocessing.StandardScaler().fit(ex_train)
ex_train_std = std_stu.transform(ex_train)
ex_test_std  = std_stu.transform(ex_test)

print('\nresult:')
bestScore = 0
for i in range(3,9):
    classifier = KNeighborsRegressor(i)
    classifier.fit(ex_train_std, de_train)
    al = classifier.score(ex_test_std, de_test)
    print('when k = {}:\nscore: {}'.format(i, al))
    if al > bestScore: 
        bestScore = al
        bestK = i
    

print('Best perfornance:\nK:{}\nscore: {}'.format(bestK, bestScore))    
