#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:22:57 2020

@author: xuebinwang
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.model_selection import cross_val_score
  

df = pd.read_csv("/Users/xuebinwang/Desktop/IE 517/hw/hw6/ccdefault.csv" )

X = df.iloc[:, 1:24].values
y = df[['DEFAULT']]

in_sample_acc = []
out_of_sample_acc = []

steps = [('scaler', StandardScaler()),
         ('decision_tree', DecisionTreeClassifier())]
pipe_part1 = Pipeline(steps)

for i in range(1,11):
    X_train, X_test, y_train, y_test =  train_test_split(X, y, 
                     test_size=0.10,
                     stratify=y,
                     random_state=i)
    pipe_part1.fit(X_train, y_train)
    y_pred_train = pipe_part1.predict(X_train)
    y_pred_test = pipe_part1.predict(X_test)
    print('Accuracy train: %.3f, Accuracy test: %.3f' %(accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test)))
    
    in_sample_acc.append(accuracy_score(y_train, y_pred_train))
    out_of_sample_acc.append(accuracy_score(y_test, y_pred_test))
    
print("The mean of the in-sample accuracy score is:",np.mean(in_sample_acc))
print("The standard deviation of the in-sample accuracy score is:",np.std(in_sample_acc))
    
print("The mean of the out-of-sample accuracy score is:",np.mean(out_of_sample_acc))
print("The standard deviation of the out-of-sample accuracy score is:",np.std(out_of_sample_acc))




scores = cross_val_score(DecisionTreeClassifier(),
                         X_train,
                         y_train,
                         cv=10)
print([float('{:.3f}'.format(i)) for i in scores])
print('CV accuracy mean for train: %.3f, Std: %.3f' % (np.mean(scores), np.std(scores)))

scores = cross_val_score(DecisionTreeClassifier(),
                         X_test,
                         y_test,
                         cv=10,
                         n_jobs=1)
print([float('{:.3f}'.format(i)) for i in scores])
print('CV accuracy: %.3f for test, Std: %.3f' % (np.mean(scores), np.std(scores)))


pipe_dt = make_pipeline(StandardScaler(),
                        DecisionTreeClassifier(criterion='gini', 
                              max_depth=4, 
                              random_state=None))

kfold = StratifiedKFold(n_splits=10,
                        random_state=1).split(X_train, y_train)


print("My name is Xuebin Wang")
print("My NetID is: xuebinw2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


