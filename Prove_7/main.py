# -*- coding: utf-8 -*-

from nnclassifier import Nnclassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets 
import pandas as pd
import numpy as np
import csv

# dataset 1
data = datasets.load_iris()
d = pd.DataFrame(data.data)
t = pd.DataFrame(data.target)

# spliting data
ex_train, ex_test, de_train, de_test = train_test_split(d, t, test_size = .35, 
                                                        random_state = 223112)

# validation data
ex_va, ex2_test, de_va, de2_test = train_test_split(ex_test, de_test, test_size = .5, 
                                                        random_state = 14)

ded = list(np.ravel(de_train))
de_va = list(np.ravel(de_va))

#setting 1
all_precent = []
best_one_1 = 0
best_precent_1 = 0
for i in range(8):
    n = Nnclassifier()
    n.setting(4, [4,3,3], 500, 0.1)
    n.valid(ex_va, de_va)
    n.fit(ex_train, ded)
    all_precent.append(n.precent)
    if n.precent[-1] > best_precent_1:
        best_precent_1 = n.precent[-1]
        best_one_1 = n
    
csvfile = "output_1.csv"   
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(all_precent)

r1 = best_one_1.predict(ex2_test,de2_test)

# setting 2
all_precent = []
best_one_2 = 0
best_precent_2 = 0
for i in range(8):
    n = Nnclassifier()
    n.setting(3, [4,5], 500, 0.2)
    n.valid(ex_va, de_va)
    n.fit(ex_train, ded)
    all_precent.append(n.precent)
    if n.precent[-1] > best_precent_2:
        best_precent_2 = n.precent[-1]
        best_one_2 = n

csvfile = "output_2.csv"   

with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(all_precent)

r2 = best_one_2.predict(ex2_test,de2_test)


        

# dataset 2
data = pd.read_csv('data_2.csv', sep = ',')



ta = pd.DataFrame(data.loc[:,'X1'])
d = data.drop(columns = ['X1'])

# splitting data
ex_train, ex_test, de_train, de_test = train_test_split(d, ta, test_size = .30, 
                                                        random_state = 212)

# validation data
ex_va, ex2_test, de_va, de2_test = train_test_split(ex_test, de_test, test_size = .5, 
                                                        random_state = 127)

ded = list(np.ravel(de_train))
de_va = list(np.ravel(de_va))
#setting 1
all_precent = []
best_one__1 = 0
best_precent__1 = 0
for i in range(5):
    n = Nnclassifier()
    n.setting(4, [16,5,5], 500, 0.2)
    n.valid(ex_va, de_va)
    n.fit(ex_train, ded)
    all_precent.append(n.precent)
    if n.precent[-1] > best_precent__1:
        best_precent__1 = n.precent[-1]
        best_one__1 = n
    
csvfile = "output__1.csv"   
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(all_precent)

r_1 = best_one__1.predict(ex2_test,de2_test)

# setting 2
all_precent = []
best_one__2 = 0
best_precent__2 = 0
for i in range(5):
    n = Nnclassifier()
    n.setting(3, [16,10], 500, 0.2)
    n.valid(ex_va, de_va)
    n.fit(ex_train, ded)
    all_precent.append(n.precent)
    if n.precent[-1] > best_precent__2:
        best_precent__2 = n.precent[-1]
        best_one__2 = n

csvfile = "output__2.csv"   

with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(all_precent)

r_2 = best_one__2.predict(ex2_test,de2_test)


count = 0

for i in range(len(r1)):
    if  list(np.ravel(de2_test))[i] == list(np.ravel(r1))[i]:
        count += 1

print(count/i)

        

