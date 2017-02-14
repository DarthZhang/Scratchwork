# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 00:26:13 2017

@author: af1tang
"""

import sys
import pickle
import os.path as path

import csv
import gzip
#import MySQLdb as mysql
#import pymysql as mysql
import pandas as pd
#import sqlalchemy
#rom sqlalchemy import create_engine
from pandas import DataFrame
from pandas.io import sql as transfer


import numpy as np
import gensim
import math
import random
import datetime
import matplotlib.pyplot as plt
import re

from scipy import stats
from collections import Counter
from itertools import combinations
from datetime import date
from datetime import time
from datetime import timedelta
from sklearn import preprocessing

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.datasets import imdb

def basic_model(dropout = 0.0, weight_constraint = 0):
    model = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu', W_constraint=weight_constraint))
    model.add(Dropout(dropout))    
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return (model)
    
def main():
    np.random.seed(7)
    data = np.loadtxt('/home/af1tang/Desktop/Practice Files/pima-indians-diabetes.csv', delimiter = ',')
    X = data[:,0:8]
    Y = data[:,8]
    
    #build model using KerasClassifier and Gridsearch
    model = KerasClassifier(build_fn=basic_model, verbose=0)
    # define the grid search parameters
    batch_size = [10, 20, 40, 60, 80, 100]
    epochs = [10, 50, 100]
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    neurons = [1, 5, 10, 15, 20, 25, 30]
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    
    param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
    #setup GridSearch w/ cross validation
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    # Fit the model
    grid_result = grid.fit(X, Y)    
    #grid_search results:
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        
    
    #otherwise, use: 
    model.fit(X, Y, nb_epoch=150, batch_size=10)
    # evaluate the model
    scores = model.evaluate(X, Y)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    predictions = model.predict(X)
    # round predictions
    predictions = [round(x[0]) for x in predictions]
    acc = accuracy_score(Y, predictions)
    f1 = f1_score (Y, predictions)
    auc = roc_auc_score (Y, predictions)
    scores = [("Accuracy", acc) , ("F1 Score", f1) , ("AUC Score",auc)]
    for s in scores:
        print("%s: %.2f" %(s[0], s[1]))

    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    cvscores = []
    for train, test in kfold.split(X, Y):
        # create model
        model = basic_model()
        # Fit the model
        model.fit(X[train], Y[train], nb_epoch=150, batch_size=10, verbose=0)
        # evaluate the model
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

def rnn_training():
    np.random.seed(7)
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
    max_review_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length, dropout=0.2))
    model.add(Dropout(0.2))
    #convolution, 1D since sequence data
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    #model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length, dropout=0.2))
    #model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, nb_epoch=3, batch_size=64)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))


if __name__ == "__main__":
    main()
    rnn_training()
