#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
#c = 10
#c = 100
#c = 1000
c = 10000
clf = SVC(kernel = 'rbf', C = c)
t0 = time()
#features_train = features_train[:len(features_train)//100]
#labels_train = labels_train[:len(labels_train)//100]
clf.fit(features_train,labels_train)
print "Training time =",time()-t0
t0 = time()
pred = clf.predict(features_test)
print "prediction time =", time() - t0
print(accuracy_score(labels_test, pred))
print(pred[10],pred[26],pred[50])
print(labels_test[10],labels_test[26],labels_test[50])
print("chris' emails =", pred[pred == 1].sum())

#########################################################


