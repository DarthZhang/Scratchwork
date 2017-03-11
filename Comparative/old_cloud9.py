# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 13:59:57 2017

@author: andy
"""

import sys
import cPickle as pickle
import os.path as path

import csv
import gzip
#import MySQLdb as mysql
#import pymysql as mysql
import pandas as pd
#import sqlalchemy
#from sqlalchemy import create_engine
from pandas import DataFrame
from pandas.io import sql as transfer


import numpy as np
import gensim
#import glove
import math
import random
import datetime
import matplotlib.pyplot as plt
import re

from scipy import stats
from collections import Counter
from itertools import combinations, tee, chain
from datetime import date
from datetime import time
from datetime import timedelta
from sklearn import preprocessing

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC as SVM
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD, RMSprop, Adam
from keras.regularizers import l1, l2

#MEMMAP
from tempfile import mkdtemp
#file_a = path.join(mkdtemp(), 'Xfiles.dat') #sentences
#file_b = path.join(mkdtemp(), 'Yfiles.dat') #x_train
#file_c = path.join(mkdtemp(), 'Zfiles.dat') #w_train
#file_d = path.join(mkdtemp(), 'AAfiles.dat') #x_test
#file_e = path.join(mkdtemp(), 'ABfiles.dat') #w_test


##### FILE LOCATIONS ######
###########################

admissions_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/ADMISSIONS.csv.gz'
diagnoses_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/DIAGNOSES_ICD.csv.gz'
icds_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/D_ICD_DIAGNOSES.csv.gz'
procedures_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/PROCEDUREEVENTS_MV.csv.gz'
labevents_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/LABEVENTS.csv.gz'
items_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/D_ITEMS.csv.gz'
labitems_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/D_LABITEMS.csv.gz'
patients_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/PATIENTS.csv.gz'

def main():
    data = modeling (admits = admits, sentences = sentences, lib = lib, dz = d)
    
##### Models #####  
def d_cnn_train(input_shape, dropout_W = 0.2, dropout_U = 0.2, optimizer = 'Adam', neurons = 100, learn_rate = .01, momentum= 0.0, W_regularizer = None, U_regularizer = None, init_mode = 'zero'):
    model = Sequential()
    model.add(Convolution1D(input_shape = input_shape, nb_filter = 300, filter_length = 3, border_mode = 'same', activation = 'relu', init = init_mode))
    model.add(MaxPooling1D(pool_length = 2))
    model.add(LSTM(output_dim=neurons, dropout_W = dropout_W, dropout_U = dropout_U, W_regularizer = W_regularizer, U_regularizer = U_regularizer))
    model.add(Dense(1, activation = 'sigmoid'))
    if optimizer == 'SGD':
        optimizer = SGD(lr = learn_rate, momentum = momentum, nesterov = True)
    elif optimizer == 'RMSprop':
        optimizer = RMSprop(lr = learn_rate)
    elif optimizer == 'Adam':
        optimizer = Adam(lr=learn_rate)
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return (model)

def d_lstm_train(input_shape, dropout_W = 0.2, dropout_U = 0.2, optimizer = 'adam', neurons = 100, learn_rate = .01, momentum= 0.0, W_regularizer = None, U_regularizer = None, init_mode = 'zero'):
    model = Sequential()
    model.add(LSTM(output_dim=neurons, dropout_W = dropout_W, dropout_U = dropout_U, W_regularizer = W_regularizer, U_regularizer = U_regularizer))
    model.add(Dense(1, activation = 'sigmoid'))
    if optimizer == 'SGD':
        optimizer = SGD(lr = learn_rate, momentum = momentum, nesterov = True)
    elif optimizer == 'RMSprop':
        optimizer = RMSprop(lr = learn_rate)
    elif optimizer == 'Adam':
        optimizer = Adam(lr=learn_rate)
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return (model)
    
def cnn_train(top_words, max_length, embedding_length, dropout_W = 0.2, dropout_U = 0.2, optimizer = 'Adam', neurons = 100, learn_rate = .01, momentum= 0.0, W_regularizer = None, U_regularizer = None, init_mode = 'zero'):
    model = Sequential()
    model.add(Embedding(top_words, embedding_length, input_length=max_length, init = init_mode))    
    model.add(Convolution1D(nb_filter = 300, filter_length = 3, border_mode = 'same', activation = 'relu'))
    model.add(MaxPooling1D(pool_length = 2))
    model.add(LSTM(output_dim=neurons, dropout_W = dropout_W, dropout_U = dropout_U, W_regularizer = W_regularizer, U_regularizer = U_regularizer))

    model.add(Dense(1, activation = 'sigmoid'))
    if optimizer == 'SGD':
        optimizer = SGD(lr = learn_rate, momentum = momentum, nesterov = True)
    elif optimizer == 'RMSprop':
        optimizer = RMSprop(lr = learn_rate)
    elif optimizer == 'Adam':
        optimizer = Adam(lr=learn_rate)
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return (model)
    
def lstm_train(top_words, max_length, embedding_length, dropout_W = 0.2, dropout_U = 0.2, optimizer = 'adam', neurons = 100, learn_rate = .01, momentum= 0.0, W_regularizer = None, U_regularizer = None, init_mode = 'zero'):

    model = Sequential()
    model.add(Embedding(top_words, embedding_length, input_length=max_length))    
    model.add(LSTM(output_dim=neurons, dropout_W = dropout_W, dropout_U = dropout_U, W_regularizer = W_regularizer, U_regularizer = U_regularizer))

    model.add(Dense(1, activation = 'sigmoid'))
    if optimizer == 'SGD':
        optimizer = SGD(lr = learn_rate, momentum = momentum, nesterov = True)
    elif optimizer == 'RMSprop':
        optimizer = RMSprop(lr = learn_rate)
    elif optimizer == 'Adam':
        optimizer = Adam(lr=learn_rate)
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return (model)

def decay(x, t_stamps, embedding_length, max_review_length):

    C = []
    print ("Making SG...")
    SG = gensim.models.Word2Vec(sentences = x, sg = 1, size = embedding_length, window = 10, min_count = 50, hs = 1, negative = 0, workers = 4)
    print ("SG embedding complete.")
    for ii in range(len(t_stamps)):
        decay_factor=np.array([math.exp(-1 * decay * elapse.total_seconds()/3600) for elapse in t_stamps[ii]])
        v = np.array(list(map(lambda x: SG[x] if x in SG.wv.vocab else [0]*embedding_length, x[ii])))
        w = np.array([np.multiply(v[index], decay_factor[index]) for index in range(len(decay_factor))])
        if w.shape[0]<max_review_length:
            temp = np.array([[0]*embedding_length for jjj in range(max_review_length-len(v))])
            w = np.concatenate((w,temp))
        elif len(w) > max_review_length:
            w = w[0:max_review_length]
        
        if ii == 0:
            W = np.memmap('w_train.mymemmap', mode = 'w+', shape = (1, w.shape[0], w.shape[1]), dtype = 'object')
            W[:] = w[:].reshape((1,max_review_length, embedding_length))
            C = w
        else:
            W= np.memmap('w_train.mymemmap', mode = 'r+', shape = (ii+1, w.shape[0], w.shape[1]), dtype = 'object')
            W[ii:,:] = w[:].reshape((1,max_review_length, embedding_length))       
            C = np.add(C, w)
    C.reshape(1, max_review_length, embedding_length)
    return (W, C)
    
##########################
    
def batch_generator(X, Y):
    X = X.astype('float32')
    Y = Y.astype('float32')
    while 1:
        for i in range(len(X)/32):
            if i%125 ==0: print('i = ' +str(i))
        yield (X[i*32:(i+1)*32], Y[i*32:(i+1)*32])     
    
    
def modeling(admits, sentences, lib, dz):

    np.random.seed(7)
    decay = .0002
    data = []; Data = []
    train = []; test = []
    keys = [k[1] for k in lib]
    
    for d in dz:
        neg = random.sample(d[1], len(d[0]))
        temp = d[0] + neg
        random.shuffle(temp)
        t1, t2 = cross_validation.train_test_split(temp, test_size = .2)
        train +=t1; test +=t2
                    

    X_train = []; t_train = []; W_train = []; Y_train = []
    X_test = []; t_test = []; W_test = []; Y_test = []
    V_train = []; V_test = []
    
    count=0
    for t in train:
        print (count)
        count+=1

        corpus = [[s[2], s[3]] for s in sentences if  (s[0] == t[0]) and (pd.to_datetime(admits[admits['HADM_ID']==s[1]].ADMITTIME.values[0]) <= t[1])]
        corpus = sorted(corpus, key = lambda x: x[1])

        corpus = list(map(list, zip(*corpus)))                  
        x_train = list(chain.from_iterable(corpus[0]))
        t_stamps = list(chain.from_iterable(corpus[1]))
        x = np.array(list(map(lambda x: keys.index(x), x_train)))
     

        temp = t_stamps[0]
        t_stamps = [ii-temp for ii in t_stamps]
                
        #append
        X_train.append(x)
        V_train.append(np.array(x_train))
        t_train.append(np.array(t_stamps))
        Y_train.append(t[3])
                
    print ("X_train made.")

    count = 0
    for t in test:
        print (count)
        count+=1
                
        corpus = [[s[2], s[3]] for s in sentences if  (s[0] == t[0]) and (pd.to_datetime(admits[admits['HADM_ID']==s[1]].ADMITTIME.values[0]) <= t[1])]
                
        corpus = sorted(corpus, key = lambda x: x[1])
        corpus = list(map(list, zip(*corpus)))                  
        x_test = list(chain.from_iterable(corpus[0]))
        t_stamps = list(chain.from_iterable(corpus[1]))
        temp = t_stamps[0]
        t_stamps = [ii-temp for ii in t_stamps]
        x = np.array(list(map(lambda x: keys.index(x), x_test)))
            
        X_test.append(x)
        V_test.append(np.array(x_train))
        t_test.append(np.array(t_stamps))
        Y_test.append(t[3])            
                           
    #training normal LSTM and CNN-LSTM          
    top_words = [9444]
    max_review_length = [1000]
    embedding_length = [300]          
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length[0])
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length[0])


    #build model using KerasClassifier and Gridsearch
    cnn = KerasClassifier(build_fn=cnn_train, verbose=1)
    lstm = KerasClassifier(build_fn=lstm_train, verbose=1)
    d_cnn = KerasClassifier(build_fn=d_cnn_train, verbose = 1)
    d_lstm = KerasClassifier(build_fn=d_lstm_train, verbose = 1)
        
    # define the grid search parameters
    batch_size = [32, 64, 128]
    epochs = [50, 100, 200]
    optimizer = ['SGD', 'RMSprop', 'Adam']
    learn_rate = (10.0**np.arange(-3,-1)).tolist()
    momentum = np.arange(.5,.9,.1).tolist()
    neurons = [50, 100, 200]
    dropout_W = [.1, .2, .5]
    dropout_U = [.1, .2, .5]
    W_regularizer = [l1(.0001), l1(.001), l1(.01), l2(.0001), l2(.001), l2(.01), None]
    U_regularizer = [l1(.0001), l1(.001), l1(.01), l2(.0001), l2(.001), l2(.01), None]
    init_mode = ['uniform', 'normal', 'zero']
    #activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    param_grid = dict(top_words=top_words, max_length = max_review_length, embedding_length = embedding_length, batch_size=batch_size, nb_epoch=epochs, optimizer = optimizer, learn_rate = learn_rate, momentum = momentum, neurons = neurons, dropout_W = dropout_W, dropout_U = dropout_U, W_regularizer = W_regularizer, U_regularizer = U_regularizer, init_mode = init_mode)
    d_param_grid = dict(input_shape = [(max_review_length[0], embedding_length[0])], batch_size=batch_size, nb_epoch=epochs, optimizer = optimizer, learn_rate = learn_rate, momentum = momentum, neurons = neurons, dropout_W = dropout_W, dropout_U = dropout_U, W_regularizer = W_regularizer, U_regularizer = U_regularizer, init_mode = init_mode)
    lr_params = {'C':(10.0**np.arange(-4,4)).tolist(), 'penalty':('l1','l2')}
    sv_params = {'C':(10.0**np.arange(-4,4)).tolist(), 'kernel':('linear', 'poly', 'rbf', 'sigmoid')}
    rf_params = {'criterion': ['gini', 'entropy']}
    
    #setup GridSearch w/ cross validation
    cnn_grid = GridSearchCV(estimator=cnn, param_grid=param_grid, scoring = 'roc_auc', cv = 5, n_jobs=-1, verbose = 100)
    lstm_grid = GridSearchCV(estimator=lstm, param_grid=param_grid, scoring = 'roc_auc', cv = 5, n_jobs=-1, verbose = 100)
    d_cnn_grid = GridSearchCV(estimator=d_cnn, param_grid=d_param_grid, scoring = 'roc_auc', cv = 5, n_jobs=-1, verbose = 100)
    d_lstm_grid = GridSearchCV(estimator=d_lstm, param_grid=d_param_grid, scoring = 'roc_auc', cv = 5, n_jobs=-1, verbose = 100)
    classics = GridSearchCV(estimator = (LR, SVM, RF), param_grid = (lr_params, sv_params, rf_params), cv = 5, scoring = 'roc_auc', n_jobs = -1, verbose = 100)

    # Fit the model
    cnn_result = cnn_grid.fit(X_train, Y_train)
    lstm_result = lstm_grid.fit(X_train, Y_train) 
    d_cnn_result = d_cnn_grid.fit(decay(x=np.array(V_train), t_stamps =t_train, embedding_length=embedding_length[0], max_review_length=max_review_length[0])[0], Y_train)
    d_lstm_result = d_lstm_grid.fit(decay(x=np.array(V_train), t_stamps =t_train, embedding_length=embedding_length[0], max_review_length=max_review_length[0])[0], Y_train) 
    classics_result = classics.fit(decay(x=V_train, t_stamps =t_train, embedding_length=embedding_length[0], max_review_length=max_review_length[0])[1], Y_train)       
       
    #grid_search results:
    print("CNN Best: %f using %s" % (cnn_result.best_score_, cnn_result.best_params_))
    means = cnn_result.cv_results_['mean_test_score']
    stds = cnn_result.cv_results_['std_test_score']
    params = cnn_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, params))
    data = {'CNN':{'means':means, 'stds':stds, 'params':params, 'best': cnn_result.best_score_, 'best_params':cnn_result.best_params_}}
        
    print("LSTM Best: %f using %s" % (lstm_result.best_score_, lstm_result.best_params_))
    means = lstm_result.cv_results_['mean_test_score']
    stds = lstm_result.cv_results_['std_test_score']
    params = lstm_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, params))
    data['LSTM'] ={'means':means, 'stds':stds, 'params':params, 'best': lstm_result.best_score_, 'best_params':lstm_result.best_params_}
    
    print("Decay CNN Best: %f using %s" % (d_cnn_result.best_score_, d_cnn_result.best_params_))
    means = d_cnn_result.cv_results_['mean_test_score']
    stds = d_cnn_result.cv_results_['std_test_score']
    params = d_cnn_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, params))
    data ['Decay CNN'] ={'means':means, 'stds':stds, 'params':params, 'best': d_cnn_result.best_score_, 'best_params':d_cnn_result.best_params_}
    
    print("Decay LSTM Best: %f using %s" % (d_lstm_result.best_score_, d_lstm_result.best_params_))
    means = d_lstm_result.cv_results_['mean_test_score']
    stds = d_lstm_result.cv_results_['std_test_score']
    params = d_lstm_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, params))       
    data['Decay LSTM']={'means':means, 'stds':stds, 'params':params, 'best': d_lstm_result.best_score_, 'best_params':d_lstm_result.best_params_}
            
    print("Best of Classics: %f using %s, %s" % (classics_result.best_score_, classics_result.best_estimator_, classics_result.best_params_))    
    means = classics_result.cv_results_['mean_test_score']
    stds = classics_result.cv_results_['std_test_score']
    params = classics_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, params))        
    data['Classics']={'means':means, 'stds':stds, 'params':params, 'best': classics_result.best_score_, 'best_params':classics_result.best_params_}
    
    Data = pd.DataFrame(data)

    return (Data)

##############################

if __name__ == '__main__':
    #from optparse import OptionParser, OptionGroup
    #desc = "Welcome to ISeeYou by af1tang."
    #version = "version 1.0"
    #opt = OptionParser (description = desc, version=version)
    #opt.add_option ('-i', action = 'store', type ='string', dest='input', help='Please input path to Database File.')
    #opt.add_option ('-o', action = 'store', type = 'string', dest='output', default='CHF_data.pickle', help='Please state desired storage file for this session.')
    #(cli, args) = opt.parse_args()
    #opt.print_help()

    with open ('/home/tangfeng/MIMIC/temp/admits.pkl', 'rb') as f:
        admits = pickle.load(f)

    with open ('/home/tangfeng/MIMIC/temp/d.pkl', 'rb') as f:
        d = pickle.load(f)
    
    with open ('/home/tangfeng/MIMIC/temp/lib.pkl', 'rb') as f:
        lib = pickle.load(f)
    
    with open ('/home/tangfeng/MIMIC/temp/sentences.pkl', 'rb') as f:
        sentences = pickle.load(f)
    

    main()  
    
###### SCRATCH WORK #########
######TESTING#######
 #       cnn = cnn_train(top_words = top_words, max_length = max_review_length, embedding_length=embedding_length)
 #       lstm = lstm_train(top_words = top_words, max_length = max_review_length, embedding_length=embedding_length)
 #           
 #       cnn.fit(X_train, Y_train, validation_split = .2, nb_epoch=100, batch_size=128, shuffle = True, verbose=1)
 #       lstm.fit(X_train, Y_train, validation_split = .2, nb_epoch=100, batch_size=128, shuffle = True, verbose=1)

 #       #testing
 #       predictions_lstm = lstm.predict_classes(X_test)
 #       predictions_cnn = cnn.predict_classes(X_test)

  #      acc = accuracy_score(Y_test, predictions_lstm)
  #      f1 = f1_score (Y_test, predictions_lstm)
  #      auc = roc_auc_score (Y_test, predictions_lstm)
  #      scores_lstm = [("Accuracy", acc) , ("F1 Score", f1) , ("AUC Score",auc)]

 #       acc = accuracy_score(Y_test, predictions_cnn)
 #       f1 = f1_score (Y_test, predictions_cnn)
 #       auc = roc_auc_score (Y_test, predictions_cnn)
 #       scores_cnn = [("Accuracy", acc) , ("F1 Score", f1) , ("AUC Score",auc)]

  #      print ("LSTM DATA: ")
  #      for s in scores_lstm:
  #          print("%s: %.2f" %(s[0], s[1]), end = " ")
  #      print ("")
  #      print ("CNN DATA: ")
  #      for s in scores_cnn:
  #          print("%s: %.2f" %(s[0], s[1]), end = " ")        
        
        
            
