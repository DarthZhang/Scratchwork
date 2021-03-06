# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 13:59:57 2017

@author: andy
"""

import sys
import pickle
#import cPickle as pickle
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
import cython
#import glove
import math
import random
import datetime
#import matplotlib.pyplot as plt
import logging
import threading
import re

from scipy import stats
from scipy.stats import uniform as sp_rand
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
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
file_a = path.join(mkdtemp(), 'Xfiles.dat') #sentences
file_b = path.join(mkdtemp(), 'Yfiles.dat') #x_train
file_c = path.join(mkdtemp(), 'Zfiles.dat') #w_train
file_d = path.join(mkdtemp(), 'AAfiles.dat') #x_test
file_e = path.join(mkdtemp(), 'ABfiles.dat') #w_test
#remember to export CUDAs...
#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
#export CUDA_HOME=/usr/local/cuda

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

    np.random.seed(7)
    options = ['classic', 'cnn', 'lstm']
    
    try:
        with open ('/home/andy/Desktop/MIMIC/temp/pretrain/x_train.pkl', 'rb') as f:
            X_train = pickle.load(f)
        with open ('/home/andy/Desktop/MIMIC/temp/pretrain/x_test.pkl', 'rb') as f:
            X_test = pickle.load(f)
        with open ('/home/andy/Desktop/MIMIC/temp/pretrain/y_train.pkl', 'rb') as f:
            Y_train = pickle.load(f)
        with open ('/home/andy/Desktop/MIMIC/temp/pretrain/y_test.pkl', 'rb') as f:
            Y_test = pickle.load(f)
        with open ('/home/andy/Desktop/MIMIC/temp/pretrain/v_train.pkl', 'rb') as f:
            V_train = pickle.load(f)
        with open ('/home/andy/Desktop/MIMIC/temp/pretrain/v_test.pkl', 'rb') as f:
            V_test = pickle.load(f)
        with open ('/home/andy/Desktop/MIMIC/temp/pretrain/t_train.pkl', 'rb') as f:
            t_train = pickle.load(f)   
        with open ('/home/andy/Desktop/MIMIC/temp/pretrain/t_test.pkl', 'rb') as f:
            t_test = pickle.load(f)
    except:
        with open ('/home/andy/Desktop/MIMIC/temp/admits.pkl', 'rb') as f:
            admits = pickle.load(f)
    
        with open ('/home/andy/Desktop/MIMIC/temp/d.pkl', 'rb') as f:
            d = pickle.load(f)
        
        with open ('/home/andy/Desktop/MIMIC/temp/lib.pkl', 'rb') as f:
            lib = pickle.load(f)
        
        with open ('/home/andy/Desktop/MIMIC/temp/sentences.pkl', 'rb') as f:
            sentences = pickle.load(f)
        print ("Splitting dataset...")
        X_train, X_test, V_train, V_test, t_train, t_test, Y_train, Y_test = get_split(admits = admits, sentences = sentences, lib = lib, dz = d)
        del sentences
    try:
        SG = gensim.models.Word2Vec.load('/home/andy/Desktop/MIMIC/temp/pretrain/SG')
        print ("Dictionary loaded.")
    except:
        print ("Making Dictionary...")
        V_train = [np.ndarray.tolist(i) for i in V_train]
        #Do NOT forget the previous step; it is very important to convert sentence to regular python list... otherwise it'll take forever.
        SG = gensim.models.Word2Vec(sentences = V_train, sg = 1, size = 300, window = 10, min_count = int(len(V_train)*.01), hs = 1, negative = 0)
        print("...saving dictionary...")
        SG.save("/home/andy/Desktop/MIMIC/temp/pretrain/SG")
    
      
    #print ("Making word vectors...")
    #W_train, C_train = decay(x= V_train, t_stamps = t_train, SG = SG)     
    #W_test, C_test = decay(x = V_test, t_stamps = t_test, SG = SG)
    #print ("Done.")        
    
    #opt = input("(1) Random or (2) Grid:    ")
    for o in options:        
        model = RandomSearch(X=X_train, Y=Y_train, V= V_train, t = t_train, SG = SG, option = o, nb_epoch = 16, cv = 3, n_iter_search = 32, jobs = 1)
        with open ("/home/andy/Desktop/MIMIC/results/randgrid_"+ str(o)+".pkl", 'wb') as f:
            pickle.dump(model.grid_scores_, f)
        with open("/home/andy/Desktop/MIMIC/results/best_params_" + str(o)+".pkl", 'wb') as f:
            pickle.dump(model.best_params_, f)
    
    #classic = RandomSearch(X=X_train, Y=Y_train, V=C_train, t=t_train, option = 'classic', cv = 3, jobs = 1)
    #with open ("/home/andy/Desktop/MIMIC/results/classic_grid.pkl", 'wb') as f:
    #    pickle.dump(classic.grid_scores_,f)
    #with open ("/home/andy/Desktop/MIMIC/results/classic_params.pkl", 'wb') as f:
    #    pickle.dump(classic.best_params_,f)
        
        #if o!='classic':
        #    X_train, X_test, V_train, V_test, t_train, t_test, Y_train, Y_test = get_split(admits = admits, sentences = sentences, lib = lib, dz = d)
        #    scores = testing(X_train = X_train, X_test=X_test, V_train=W_train, V_test=W_test, t_train=t_train, t_test=t_test, Y_train=Y_train, Y_test = Y_test, preset=model.best_params_, option= o, nb_epoch = 50)

       #     data = pd.DataFrame({'best_hyperparam': model.best_params_, 'acc': scores['acc'],
        #                         'f1': scores['f1'], 'auc':scores['auc']})
                                 
         #   with open ("/home/tangfeng/MIMIC/results/data_"+str(o)+".pkl", 'wb') as f:
          #      pickle.dump(data, f)
    

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
    
##### BATCH THREAD SAFE GENERATORS #####
class threadsafe_iter:
    def __init__(self,it):
        self.it = it
        self.lock = threading.Lock();
    
    def __iter__(self):
        return (self)
    
    def next(self):
        with self.lock:
            return (self.it.next())

def threadsafe_generator(f):
    def g(*a, **kw):
        return (threadsafe_iter(f(*a, **kw)))
    return (g)

def decay_generator(x, y, t_stamps, embedding_length=300, max_review_length=1000, SG=0):
    lst = []
    #for i in xrange(0,len(x), 128):
    for i in range(0, len(x), 128):
        lst.append(i)
    lst.append(len(x))
    
    decay = .0002
    if SG ==0:
        print ("dictionary not defined")
        return ([])
        
    while True:
        W = []
        for ii in range (len(x)):
            #print (ii)
            decay_factor=np.array([math.exp(-1 * decay * elapse.total_seconds()/3600) for elapse in t_stamps[ii]])
            v = np.array(list(map(lambda x: SG[x] if x in SG.wv.vocab else [0]*embedding_length, x[ii])))
            w = np.array([np.multiply(v[index], decay_factor[index]) for index in range(len(decay_factor))])
            if w.shape[0]<max_review_length:
                temp = np.array([[0]*embedding_length for jjj in range(max_review_length-len(v))])
                w = np.concatenate((w,temp))
            elif len(w) > max_review_length:
                w = w[0:max_review_length]
            W.append(w)
            if (ii+1 in lst) and ii>0:
                yield(np.array(W), y[lst[lst.index(ii+1)-1]:lst[lst.index(ii+1)]])
                W= []

        
def decay_norm (x, t_stamps, embedding_length=300, max_review_length=1000, SG=0):
    decay = .0002
    if SG ==0:
        print ("dictionary not defined")
        return ([])
    for ii in range (len(t_stamps)):
        print (ii)
        decay_factor=np.array([math.exp(-1 * decay * elapse.total_seconds()/3600) for elapse in t_stamps[ii]])
        v = np.array(list(map(lambda x: SG[x] if x in SG.wv.vocab else [0]*embedding_length, x[ii])))
        w = np.array([np.multiply(v[index], decay_factor[index]) for index in range(len(decay_factor))])
        if w.shape[0]<max_review_length:
            temp = np.array([[0]*embedding_length for jjj in range(max_review_length-len(v))])
            w = np.concatenate((w,temp))
        elif len(w) > max_review_length:
            w = w[0:max_review_length]
        if ii == 0:
            C = w
        else:
            C = np.add(C,w)
    return (C)
        
    
##########################
def get_split(admits, sentences, lib, dz):
    train = []; test = []
    keys = [k[1] for k in lib]
    
    for d in dz:
        neg = random.sample(d[1], len(d[0]))
        temp = d[0] + neg
        random.shuffle(temp)
        t1, t2 = cross_validation.train_test_split(temp, test_size = .2)
        train +=t1; test +=t2
        
    X_train = []; t_train = []; Y_train = []
    X_test = []; t_test = [];  Y_test = []
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
    return (X_train, X_test, V_train, V_test, t_train, t_test, Y_train, Y_test)
        
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
### Random Search #####            
def RandomSearch(X=[], Y = [], V = [], t = [], SG = 0, top_words = 9444, max_review_length = 1000, embedding_length = 300, batch_size = 128, nb_epoch =10, option = 'cnn', cv=2, n_iter_search = 20, jobs = -1):
    lr_params = {'C':sp_rand(.0001, 1000), 'penalty':('l1','l2')}
    sv_params = {'C':sp_rand(.0001,1000), 'kernel':('linear', 'poly', 'rbf', 'sigmoid')}
    rf_params = {'criterion': ['gini', 'entropy']}
    
    optimizer = ['SGD', 'RMSprop', 'Adam']
    learn_rate = sp_rand(.0001, .01)
    momentum = sp_rand(.5, .9)
    #neurons = [100]
    dropout_W = sp_rand(0, .5)
    dropout_U = sp_rand(0, .5)
    W_regularizer = [l1(.0001), l1(.001), l1(.01), l2(.0001), l2(.001), l2(.01), None]
    U_regularizer = [l1(.0001), l1(.001), l1(.01), l2(.0001), l2(.001), l2(.01), None]
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    params = dict(optimizer = optimizer, learn_rate = learn_rate, momentum = momentum, W_regularizer = W_regularizer, U_regularizer = U_regularizer, dropout_W = dropout_W, dropout_U = dropout_U, init_mode = init_mode)
    
    X_train = sequence.pad_sequences(X, maxlen=max_review_length)
    
    if option == 'classic':    
        grid = RandomizedSearchCV(estimator = (LR, SVM, RF), param_distributions = (lr_params, sv_params, rf_params), scoring = 'roc_auc', n_jobs = jobs, n_iter=n_iter_search, verbose = 1)
        results = grid.fit(decay_norm(x=np.array(V), t_stamps =t, embedding_length=embedding_length, max_review_length=max_review_length, SG = SG), Y)       
        #results = grid.fit(V,Y)
    elif option == 'cnn':
        model = KerasClassifier(build_fn=cnn_train, top_words=top_words, max_length = max_review_length, embedding_length = embedding_length, batch_size = batch_size, nb_epoch = nb_epoch, verbose=1)
        grid = RandomizedSearchCV(estimator=model, param_distributions=params,  cv = cv, n_jobs=jobs, n_iter=n_iter_search, verbose = 1)
        results = grid.fit(X_train,Y)
        
    elif option == 'lstm':
        model = KerasClassifier(build_fn=lstm_train, top_words=top_words, max_length = max_review_length, embedding_length = embedding_length, batch_size = batch_size, nb_epoch = nb_epoch, verbose=1)
        grid = RandomizedSearchCV(estimator=model, param_distributions=params,  cv = cv, n_jobs=jobs, n_iter=n_iter_search, verbose = 1)
        results = grid.fit(X_train,Y)
        
    elif option == 'd_cnn':
        model = KerasClassifier(build_fn=d_cnn_train, batch_size = batch_size, nb_epoch = nb_epoch, verbose = 1)
        grid = RandomizedSearchCV(estimator=model, param_distributions=params, cv = cv, n_jobs=jobs, n_iter=n_iter_search, verbose = 1)
        results = grid.fit(decay_generator(x=np.array(V), y = Y, t_stamps =t, embedding_length=embedding_length, max_review_length=max_review_length, SG = SG), Y)
        #results = grid.fit(V, Y)
        
    elif option == 'd_lstm':
        model = KerasClassifier(build_fn=d_lstm_train, batch_size = batch_size, nb_epoch = nb_epoch, verbose = 1)
        grid = RandomizedSearchCV(estimator=model, param_distributions=params, cv = cv, n_jobs=jobs, n_iter=n_iter_search, verbose = 1)
        results = grid.fit(decay_generator(x=np.array(V), y = Y, t_stamps =t, embedding_length=embedding_length, max_review_length=max_review_length, SG = SG), Y)
        #results = grid.fit(V,Y)
    report(results.cv_results_)
    return (results)
    
    
### Grid Search ###
    
def classic_modeling (V, t, Y, SG, max_review_length = 1000, embedding_length = 300):
    lr_params = {'C':(10.0**np.arange(-4,4)).tolist(), 'penalty':('l1','l2')}
    sv_params = {'C':(10.0**np.arange(-4,4)).tolist(), 'kernel':('linear', 'poly', 'rbf', 'sigmoid')}
    rf_params = {'criterion': ['gini', 'entropy']}
    
    grid = GridSearchCV(estimator = (LR, SVM, RF), param_grid = (lr_params, sv_params, rf_params), scoring = 'roc_auc', n_jobs = -1, verbose = 1)
    classics_result = grid.fit(decay_norm(x=np.array(V), t_stamps =t, embedding_length=embedding_length, max_review_length=max_review_length, SG = SG), Y)       
    #classics_result = grid.fit(V, Y)
    report(classics_result.cv_results_)
    return (classics_result)

def grid_search (x, y, v, t, SG, top_words = 9444, max_review_length=1000, embedding_length =300, batch_size = 128, nb_epoch=16, n_jobs = 1, option = 'd_cnn', param_grid = {}, preset = {}):       
    x = sequence.pad_sequences (x, maxlen=max_review_length)
    if option == 'cnn':
        preset.update({'build_fn':cnn_train, 'top_words':top_words, 'max_length':max_review_length, 'embedding_length': embedding_length, 'batch_size': batch_size, 'nb_epoch':nb_epoch, 'verbose':1})
        model = KerasClassifier(**preset)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv = 3, n_jobs=n_jobs, verbose = 1)
        grid_result = grid.fit(x, y)
        report(grid_result.cv_results_)
        return (grid_result)

    elif option == 'lstm':
        preset.update({'build_fn':lstm_train, 'top_words':top_words, 'max_length':max_review_length, 'embedding_length': embedding_length, 'batch_size': batch_size, 'nb_epoch':nb_epoch, 'verbose':1})
        model = KerasClassifier(**preset)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv = 3, n_jobs=n_jobs, verbose = 1)
        grid_result = grid.fit(x, y)
        report(grid_result.cv_results_)
        return (grid_result)

    data = []
    x = np.array(x)     #convert to numpy form before splitting
    y = np.array(y)
    t = np.array(t)
    #for key, value in param_grid.iteritems():
    for key, value in param_grid.items():
        for kk in value:
            print (key, kk)
            preset.update({'input_shape': (max_review_length, embedding_length), key:kk})
            if option == 'd_cnn':
                model = d_cnn_train(**preset)
            elif option == 'd_lstm':
                model = d_lstm_train(**preset)
            skf = StratifiedKFold (n_splits = 3, shuffle = True, random_state = 8)
            cvscore = []
            for train, test in skf.split(x, y):
                x_train, x_test = x[train], x[test]
                y_train, y_test = y[train], y[test]
                t_train, t_test = t[train], t[test]
                model.fit_generator(decay_generator(x = x_train, y = y_train, t_stamps = t_train, SG = SG), samples_per_epoch = len(x_train), nb_epoch = nb_epoch, nb_worker=n_jobs)
                score = model.evaluate_generator(decay_generator(x = x_test, y = y_test, t_stamps = t_test, SG = SG), val_samples = len(x_test), nb_worker = n_jobs)
                print("%s: %.2f%%" % model.metrics_names[1], score*100)
                cvscore.append(score[1]*100)
            temp = {'model':option, key:kk, 'mean_score': np.mean(cvscore), 'std': np.std(cvscore)}
            data.append(temp)   
    return (data)

def testing(X_train=[], X_test=[], V_train=[], V_test=[], t_train=[], t_test=[], Y_train=[], Y_test=[], top_words = 9444, max_review_length = 1000, embedding_length = 300, batch_size = 128, nb_epoch =100, preset = {}, option = 'lstm'):
    X_train = sequence.pad_sequences(X_train, maxlen = max_review_length)    
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)    
    if option == 'cnn':
        preset.update({'build_fn':cnn_train, 'top_words':top_words, 'max_length':max_review_length, 'embedding_length': embedding_length, 'batch_size': batch_size, 'nb_epoch':nb_epoch, 'verbose':1})
        model = KerasClassifier(**preset)
    elif option == 'lstm':
        preset.update({'build_fn':lstm_train, 'top_words':top_words, 'max_length':max_review_length, 'embedding_length': embedding_length, 'batch_size': batch_size, 'nb_epoch':nb_epoch, 'verbose':1})
        model = KerasClassifier(**preset)
    elif option == 'd_cnn':
        preset.update({'build_fn':d_cnn_train,'batch_size': batch_size, 'nb_epoch':nb_epoch, 'verbose':1})
        model = KerasClassifier(**preset)
    elif option == 'd_lstm':
        preset.update({'build_fn':d_lstm_train,'batch_size': batch_size, 'nb_epoch':nb_epoch, 'verbose':1})
        model = KerasClassifier(**preset)
    else: 
        print("ERROR AT TRAINING PHASE OF TESTING.")
    
    if option == 'cnn' or option == 'lstm':
        model.fit(X_train,Y_train)
    elif option == 'classic':
        model.fit(decay_norm(x=np.array(V_train), t_stamps =t_train, embedding_length=embedding_length, max_review_length=max_review_length)[0], Y_train)

    predict =model.predict(X_test)
    acc = accuracy_score(Y_test, predict)
    f1 = f1_score(Y_test, predict)
    auc = roc_auc_score(Y_test, predict)
    return ({'acc': acc, 'f1':f1, 'auc':auc})
    
##############################

if __name__ == '__main__':
   # from optparse import OptionParser, OptionGroup
   # desc = "Welcome to PipeLiner by af1tang."
   # version = "version 1.0"
   # opt = OptionParser (description = desc, version=version)
   # opt.add_option ('-i', action = 'store', type ='string', dest='input', help='Please input path to Database File.')
   # opt.add_option ('-o', action = 'store', type = 'string', dest='output', default='CHF_data.pickle', help='Please state desired storage file for this session.')
   # (cli, args) = opt.parse_args()
   # opt.print_help()

    print ("+++++++++++")

        
    main()  

##### SCRATCH WORK #####

   #     if ii == 0:
   #         W = np.memmap(file_a, mode = 'w+', shape = (1, w.shape[0], w.shape[1]), dtype = 'object')
   #         W[:] = w[:].reshape((1,max_review_length, embedding_length))
   #         C = w
   #     else:
   #         W= np.memmap(file_a, mode = 'r+', shape = (ii+1, w.shape[0], w.shape[1]), dtype = 'object')
   #         W[ii:,:] = w[:].reshape((1,max_review_length, embedding_length))       
   #         C = np.add(C, w)
    #C.reshape(1, max_review_length, embedding_length)
    #return (W, C)

'''
    
def deep_modeling(X, Y, V, t, SG, top_words = 9444, max_review_length = 1000, embedding_length = 300, batch_size = 128, nb_epoch =100, option = 'cnn', grid_option='init_mode', preset = None):
    
    # define the grid search parameters
    optimizer = ['SGD', 'RMSprop', 'Adam']
    learn_rate = (10.0**np.arange(-3,-1)).tolist()
    momentum = np.arange(.5, .9, .1).tolist()
    #neurons = [100]
    dropout_W = np.arange(.1, .5, .1).tolist()
    dropout_U = np.arange(.1, .5, .1).tolist()
    W_regularizer = [l1(.0001), l1(.001), l1(.01), l2(.0001), l2(.001), l2(.01), None]
    U_regularizer = [l1(.0001), l1(.001), l1(.01), l2(.0001), l2(.001), l2(.01), None]
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    #activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    if grid_option == 'optimizer':
        param_grid = dict(optimizer = optimizer)
        #d_param_grid = dict(optimizer = optimizer)
    elif grid_option == 'learn':
        param_grid = dict(learn_rate = learn_rate, momentum = momentum)
    elif grid_option == 'init_mode':
        param_grid = dict(init_mode = init_mode)
    elif grid_option == 'regularizer':
        param_grid = dict(W_regularizer = W_regularizer, U_regularizer = U_regularizer)
    elif grid_option == 'dropout':
        param_grid = dict(dropout_W = dropout_W, dropout_U = dropout_U)
    else:
        print ("Error with GridSearch Option Input...")         
         
    #training normal LSTM and CNN-LSTM                 
    X_train = sequence.pad_sequences(X, maxlen=max_review_length)

    #build model using KerasClassifier and Gridsearch
    if option == 'cnn':
        if len(preset)<1:
            model = KerasClassifier(build_fn=cnn_train, top_words=top_words, max_length = max_review_length, embedding_length = embedding_length, batch_size = batch_size, nb_epoch = nb_epoch, verbose=1)
        else:            
            preset.update({'build_fn':cnn_train, 'top_words':top_words, 'max_length':max_review_length, 'embedding_length':embedding_length, 'batch_size': batch_size, 'nb_epoch':nb_epoch, 'verbose':1})
            model = KerasClassifier(**preset)
        
    elif option == 'lstm':
        if len(preset)<1:
            model = KerasClassifier(build_fn=lstm_train, top_words=top_words, max_length = max_review_length, embedding_length = embedding_length, batch_size = batch_size, nb_epoch = nb_epoch, verbose=1)
        else:
            preset.update({'build_fn':lstm_train, 'top_words':top_words, 'max_length':max_review_length, 'embedding_length':embedding_length, 'batch_size': batch_size, 'nb_epoch':nb_epoch, 'verbose':1})            
            model = KerasClassifier(**preset)

    elif option == 'd_cnn':
        if len(preset)<1:
            model = KerasClassifier(build_fn=d_cnn_train, batch_size = batch_size, nb_epoch = nb_epoch, verbose = 1)
        else:
            preset.update({'build_fn':d_cnn_train, 'batch_size': batch_size, 'nb_epoch': nb_epoch, 'verbose':1})
            model = KerasClassifier(**preset)
            
    elif option == 'd_lstm':
        if len(preset)<1:
            model = KerasClassifier(build_fn=d_lstm_train, batch_size = batch_size, nb_epoch = nb_epoch, verbose = 1)
        else:
            preset.update({'build_fn':d_lstm_train, 'batch_size': batch_size, 'nb_epoch': nb_epoch, 'verbose':1})
            model = KerasClassifier(**preset)
 

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv = 5, n_jobs=-1, verbose = 1)

    # Fit the model
    if option == 'cnn' or option == 'lstm':
        grid_result = grid.fit(X_train,Y)
    elif option == 'classic':
        grid_result = grid.fit(decay_generator(x=np.array(V), t_stamps =t, embedding_length=embedding_length, max_review_length=max_review_length, SG = SG)[0], Y)

    report(grid_result.cv_results_)
    return (grid_result)
'''
