# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:13:35 2017

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
import tensorflow as tf
#import sqlalchemy
#from sqlalchemy import create_engine
from pandas import DataFrame
from pandas.io import sql as transfer


import numpy as np
import gensim
#import cython
#import glove
import math
import random
import datetime
import time as TIME
#import matplotlib.pyplot as plt
import logging
import threading
import re

from scipy import stats
from scipy.stats import uniform as sp_rand
from scipy.stats import randint as sp_randint
from collections import Counter
from itertools import combinations, tee, chain
from datetime import date
from datetime import time
from datetime import timedelta
from sklearn import preprocessing

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier

from keras.preprocessing import sequence
#from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, merge
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

    #'/home/andy/Desktop/MIMIC/temp/pretrain/...'
    try:
        with open ('/home/andy/Desktop/MIMIC/results/random_rf.pkl', 'rb') as f:
            rf = pickle.load(f)
    
        with open ('/home/andy/Desktop/MIMIC/results/random_svm.pkl', 'rb') as f:
            svm = pickle.load(f)
    
        with open ('/home/andy/Desktop/MIMIC/results/random_lr.pkl', 'rb') as f:
            lr = pickle.load(f)
    
        with open ('/home/andy/Desktop/MIMIC/results/random_lstm.pkl', 'rb') as f:
            lstm = pickle.load(f)
    
        with open ('/home/andy/Desktop/MIMIC/results/random_cnn.pkl', 'rb') as f:
            cnn = pickle.load(f)
    
        with open ('/home/andy/Desktop/MIMIC/results/random2_d_cnn.pkl', 'rb') as f:
            d_cnn = pickle.load(f)
    
        with open ('/home/andy/Desktop/MIMIC/results/random2_d_lstm.pkl', 'rb') as f:
            d_lstm = pickle.load(f)
    
        rf = pd.DataFrame([pd.Series(i) for i in rf])
        svm = pd.DataFrame([pd.Series(i) for i in svm])
        lr = pd.DataFrame([pd.Series(i) for i in lr])
        lstm = pd.DataFrame([pd.Series(i) for i in lstm])
        cnn = pd.DataFrame([pd.Series(i) for i in cnn])
        d_cnn = pd.DataFrame([pd.Series(i) for i in d_cnn])
        d_lstm = pd.DataFrame([pd.Series(i) for i in d_lstm])
        dfs = [d_lstm, d_cnn, cnn, lstm, rf, svm, lr]
        
    except:
        return ("Error.")
    
    Data = []
        
    np.random.seed(8)
    for df in dfs:
        temp = df.sort_values('mean_score', ascending=False)
        settings = temp.iloc[0].to_dict()
        if settings['model'] == 'lr':
            param_grid = {'C':settings['C'], 'penalty': settings['penalty']}
        elif settings['model'] == 'svm':
            param_grid = {'C':settings['C']}
        elif settings['model'] == 'rf':
            param_grid = {'criterion': settings['criterion'], 'n_estimators': settings['n_estimators'], 'bootstrap': settings['boostrap']}
        else:
            param_grid= dict(optimizer = settings['optimizer'], learn_rate = settings['learn_rate'], momentum = settings['momentum'],  dropout_W = settings['dropout_W'], dropout_U = settings['dropout_U'], init_mode = settings['init_mode'])
            
        if settings['model'] == 'd_lstm' or settings['model'] == 'd_cnn':
            trainable = True
        else:
            trainable = False
                
        t1 = TIME.time()
        data = testing(option = settings['model'], nb_epoch = 100, cv = 3, n_jobs = 1, param_grid = param_grid, decay = settings['decay'], trainable = trainable)    
        t2 = TIME.time()
        with open ("/home/andy/Desktop/MIMIC/results/test_"+ str(settings['model'])+".pkl", 'wb') as f:
            pickle.dump(data, f)
        print ("Pickle successful!")
        print ("Training completed in "+str((t2-t1)/3600) + " hours")
            
    Data += data
    with open ("/home/andy/Desktop/MIMIC/results/testing_results.pkl", 'wb') as f:
        pickle.dump(Data, f)
    print ("Done.")

def get_testing_splits():
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
    print ("Making Dictionary...")

    del sentences
    V_train = [np.ndarray.tolist(i) for i in V_train]
    SG = gensim.models.Word2Vec(sentences = V_train, sg = 1, size = 300, window = 10, hs = 1, negative = 0) 
    weights = SG.wv.syn0
    vocab = dict([(k, v.index) for k, v in SG.wv.vocab.items()])
    w2i, i2w = vocab_index(vocab)
    w_train = [list(map(lambda i: w2i[i] if i in w2i.keys() else 0, vv)) for vv in V_train] 
    #w_test = [list(map(lambda i: w2i[i] if i in w2i.keys() else 0, vv) for vv in V_test)]
    
    return (weights, X_train, X_test, w_train, V_test, t_train, t_test, Y_train, Y_test)

##### Models #####  
def vocab_index (vocab):
    word2idx = vocab
    idx2word = dict([(v,k) for k, v in vocab.items()])
    return (word2idx, idx2word)

def embedding_layer (weights):
    layer = Embedding (input_dim = weights.shape[0], output_dim = weights.shape[1], weights = [weights], trainable = False)
    return (layer)
    
def cnn_train(top_words, max_length, embedding_length, dropout_W = 0.2, dropout_U = 0.2, optimizer = 'Adam', neurons = 100, learn_rate = .01, momentum= 0.0, W_regularizer = None, U_regularizer = None, init_mode = 'zero', trainable = True, weights = {}):

    if trainable == True:
        model = Sequential()
        model.add(Embedding(top_words, embedding_length, input_length=max_length, init = init_mode))  
        model.add(Convolution1D(nb_filter = 300, filter_length = 3, border_mode = 'same', activation = 'relu'))
        model.add(MaxPooling1D(pool_length = 2))
        model.add(LSTM(output_dim=neurons, dropout_W = dropout_W, dropout_U = dropout_U, W_regularizer = W_regularizer, U_regularizer = U_regularizer, init = init_mode))
        
        model.add(Dense(1, activation = 'sigmoid'))
        if optimizer == 'SGD':
            optimizer = SGD(lr = learn_rate, momentum = momentum, nesterov = True)
        elif optimizer == 'RMSprop':
            optimizer = RMSprop(lr = learn_rate)
        elif optimizer == 'Adam':
            optimizer = Adam(lr=learn_rate)
        model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
        return (model)
        
    else:
        x = Input(shape = (max_length,) , dtype = 'int32', name = 'x')
        embed = embedding_layer(weights)(x)
        decay = Input(shape = (max_length, 1) , name = 'decay')
        xx = merge([embed, decay], mode = lambda x: tf.mul(x[0], x[1]), output_shape = (max_length, weights.shape[1]))

        xx = Convolution1D(nb_filter = 300, filter_length = 3, border_mode = 'same', activation = 'relu') (xx)
        xx = MaxPooling1D(pool_length = 2)(xx)
        xx = LSTM(output_dim=neurons, dropout_W = dropout_W, dropout_U = dropout_U, W_regularizer = W_regularizer, U_regularizer = U_regularizer, init = init_mode) (xx)
        y = Dense(1, activation = 'sigmoid') (xx)
          
        model = Model(input = [x, decay], output = [y])      
        if optimizer == 'SGD':
            optimizer = SGD(lr = learn_rate, momentum = momentum, nesterov = True)
        elif optimizer == 'RMSprop':
            optimizer = RMSprop(lr = learn_rate)
        elif optimizer == 'Adam':
            optimizer = Adam(lr=learn_rate)
        model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
        return (model)
        

    
def lstm_train(top_words, max_length, embedding_length, dropout_W = 0.2, dropout_U = 0.2, optimizer = 'adam', neurons = 100, learn_rate = .01, momentum= 0.0, W_regularizer = None, U_regularizer = None, init_mode = 'zero', trainable = True, weights = {}, t=[]):

    if trainable == True:
        model = Sequential()
        model.add(Embedding(top_words, embedding_length, input_length=max_length, init = init_mode))  
        model.add(LSTM(output_dim=neurons, dropout_W = dropout_W, dropout_U = dropout_U, W_regularizer = W_regularizer, U_regularizer = U_regularizer, init = init_mode))
        model.add(Dense(1, activation = 'sigmoid'))
        if optimizer == 'SGD':
            optimizer = SGD(lr = learn_rate, momentum = momentum, nesterov = True)
        elif optimizer == 'RMSprop':
            optimizer = RMSprop(lr = learn_rate)
        elif optimizer == 'Adam':
            optimizer = Adam(lr=learn_rate)
        model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
        return (model)
    
    else:
        x = Input(shape = (max_length,) , dtype = 'int32', name = 'x')
        embed = embedding_layer(weights)(x)
        decay = Input(shape = (max_length, 1) , name = 'decay')

        xx = merge([embed, decay], mode = lambda x: tf.mul(x[0], x[1]), output_shape = (max_length, weights.shape[1]))

        xx = LSTM(output_dim=neurons, dropout_W = dropout_W, dropout_U = dropout_U, W_regularizer = W_regularizer, U_regularizer = U_regularizer, init = init_mode) (xx)
        y = Dense(1, activation = 'sigmoid') (xx)        
        
        model = Model(input = [x, decay], output = [y])        
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

def decay_generator(x, y, t_stamps, embedding_length=300, max_review_length=1000, decay = .00001, SG=0):
    lst = []
    #for i in xrange(0,len(x), 128):
    for i in range(0, len(x), 128):
        lst.append(i)
    lst.append(len(x))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    if SG ==0:
        print ("dictionary not defined")
        
    while True:
        W = []
        for ii in range (len(x)):
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

        
def decay_norm (x, t_stamps, embedding_length=300, max_review_length=1000, decay = 0.00001, SG=0):

    if SG ==0:
        print ("dictionary not defined")
        return ([])
    for ii in range (len(t_stamps)):
        decay_factor=np.array([math.exp(-1 * decay * elapse.total_seconds()/3600) for elapse in t_stamps[ii]])
        v = np.array(list(map(lambda x: SG[x] if x in SG.wv.vocab else [0]*embedding_length, x[ii])))
        w = np.array([np.multiply(v[index], decay_factor[index]) for index in range(len(decay_factor))])
        w = w.sum(axis = 0)
        w = w.reshape(1, len(w))
        if ii == 0:
            C = w
        else:
            C = np.concatenate((C,w), axis = 0)
    print ("C: {0}".format(C.shape))
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
        V_test.append(np.array(x_test))
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
            
def testing(top_words = 9444, max_review_length = 1000, embedding_length = 300, batch_size = 128, nb_epoch =100, cv = 3, n_jobs = 1, param_grid = {}, option = 'lstm', decay = 0.0, trainable = False):

    data = []
    
    for sess in range(cv):    
        accs = []
        f1s = []
        aucs = []
        preset = {}
        print ("Session: {0}".format(sess))
        weights, X_train, X_test, V_train, V_test, t_train, t_test, Y_train, Y_test = get_testing_splits()
        X_train = sequence.pad_sequences(X_train, maxlen = max_review_length)    
        X_test = sequence.pad_sequences(X_test, maxlen=max_review_length) 
        V_train = sequence.pad_sequences(V_train, maxlen= max_review_length)
        Y_train = np.array(Y_train)
        Y_test = np.array(Y_test)
        t_train = np.array(t_train)
        t_test = np.array(t_test)
    
        decay_factors=[[math.exp(-1 * decay * elapse.total_seconds()/3600) for elapse in tt] for tt in t_train]
        decay_factors = sequence.pad_sequences(decay_factors, maxlen=max_review_length)
        shape = decay_factors.shape
        decay_factors = decay_factors.reshape(shape[0], shape[1], 1)
        
        for key, value in param_grid.items():
            preset.update({key:value})
            
        if option == 'lstm' or option == 'd_lstm':
            if trainable == True:
                preset.update({'top_words':top_words, 'max_length':max_review_length, 'embedding_length': embedding_length})
            else:
                preset.update({'top_words':weights.shape[0], 'max_length':max_review_length, 'embedding_length': weights.shape[1]}, trainable = False, weights = weights)
            model = lstm_train(**preset)
            classic = False
        elif option == 'cnn' or option == 'd_cnn':
            if trainable == True:
                preset.update({'top_words':top_words, 'max_length':max_review_length, 'embedding_length': embedding_length}, trainable = True)
            else:
                preset.update({'top_words':weights.shape[0], 'max_length':max_review_length, 'embedding_length': weights.shape[1]}, trainable = False, weights = weights)
            model = cnn_train(**preset)
            classic = False
        elif option == 'lr':
            preset.update({'verbose':1})
            model = LogisticRegression(**preset)
            classic = True
        elif option == 'svm':
            preset.update({'verbose':True})
            model = SVC(**preset)
            classic = True
        elif option == 'rf':
            preset.update({'verbose':1})
            model = RandomForestClassifier(**preset)
            classic = True
        print (preset)
    
        if classic == True:
            model.fit(decay_norm(x=V_train, t_stamps = t_train, decay = decay), Y_train)
            predict =model.predict(X_test)
            acc = accuracy_score(Y_test, predict)
            f1 = f1_score(Y_test, predict)
            auc = roc_auc_score(Y_test, predict)
            
            print("%s: %.2f%%, %s: %.2f%%, %s, %.2f%%" % ("accuracy", acc*100, "f1", f1*100, "auc", auc*100))
            accs.append(acc) 
            f1s.append(f1)
            aucs.append(auc)
        elif trainable == True:
            model.fit(X_train, Y_train, batch_size = batch_size, nb_epoch = nb_epoch, verbose = 1)
            acc = model.evaluate(X_test, Y_test, batch_size = batch_size, verbose = 1)
            predict = model.predict(X_test)
            f1 = f1_score(Y_test, predict)
            auc = roc_auc_score(Y_test, predict)
            
            print("%s: %.2f%%, %s: %.2f%%, %s, %.2f%%" % (model.metrics_names[1], acc[1]*100, "f1", f1*100, "auc", auc*100))         
            accs.append(acc[1]) 
            f1s.append(f1)
            aucs.append(auc)           
        else:
            model.fit(x = [V_train, decay_factors], y = Y_train, batch_size = batch_size, nb_epoch = nb_epoch, verbose = 1)
            acc = model.evaluate(X_test, Y_test, batch_size = batch_size, verbose = 1)
            predict = model.predict(X_test)
            f1 = f1_score(Y_test, predict)
            auc = roc_auc_score(Y_test, predict)
            
            print("%s: %.2f%%, %s: %.2f%%, %s, %.2f%%" % (model.metrics_names[1], acc[1]*100, "f1", f1*100, "auc", auc*100))         
            accs.append(acc[1]) 
            f1s.append(f1)
            aucs.append(auc)             
    data = {'model':option, 'decay':decay}
    data.update(preset)
    data.update({'mean_acc': np.mean(accs), 'std_acc': np.std(accs)})
    data.update({'mean_auc': np.mean(aucs), 'std_auc': np.std(aucs)})
    data.update({'mean_f1': np.mean(f1s), 'std_f1': np.std(f1s)})
    return (data)  

### Random Search #####            
def random_search (x, y, v, t, weights, top_words = 9444, max_review_length=1000, embedding_length =300, batch_size = 128, nb_epoch=16, cv=3, n_jobs = 1, option = 'd_cnn', param_grid = {}, preset = {}, n_iter = 40, trainable = False):
    x = sequence.pad_sequences (x, maxlen=max_review_length)
    v = sequence.pad_sequences (v, maxlen=max_review_length)
    data = []
    x = np.array(x)     #convert to numpy form before splitting
    v = np.array(v)
    y = np.array(y)
    t = np.array(t)
    
    for sess in range(n_iter):
        print ("Session: {0}".format(sess))
        for key, value in param_grid.items():
            try:
                preset.update({key:random.choice(value)})
            except:
                preset.update({key:value.rvs(1)[0]})
                
        decay = sp_rand(0, 0.00001).rvs(1)[0]
        decay_factors=[[math.exp(-1 * decay * elapse.total_seconds()/3600) for elapse in tt] for tt in t]
        decay_factors = sequence.pad_sequences(decay_factors, maxlen=max_review_length)
        shape = decay_factors.shape
        decay_factors = decay_factors.reshape(shape[0], shape[1], 1)
                
        if option == 'lstm' or option == 'd_lstm':
            if trainable == True:
                preset.update({'top_words':top_words, 'max_length':max_review_length, 'embedding_length': embedding_length})
            else:
                preset.update({'top_words':weights.shape[0], 'max_length':max_review_length, 'embedding_length': weights.shape[1]}, trainable = False, weights = weights)
            model = lstm_train(**preset)
            classic = False
        elif option == 'cnn' or option == 'd_cnn':
            if trainable == True:
                preset.update({'top_words':top_words, 'max_length':max_review_length, 'embedding_length': embedding_length}, trainable = True)
            else:
                preset.update({'top_words':weights.shape[0], 'max_length':max_review_length, 'embedding_length': weights.shape[1]}, trainable = False, weights = weights)
            model = cnn_train(**preset)
            classic = False
        elif option == 'lr':
            preset.update({'verbose':1})
            model = LogisticRegression(**preset)
            classic = True
        elif option == 'svm':
            preset.update({'verbose':True})
            model = SVC(**preset)
            classic = True
        elif option == 'rf':
            preset.update({'verbose':1})
            model = RandomForestClassifier(**preset)
            classic = True
            
        print (preset)

        
        skf = StratifiedKFold (n_splits = cv, shuffle = True, random_state = 8)
        cvscore = []
        for train, test in skf.split(x, y):
            x_train, x_test = x[train], x[test]
            y_train, y_test = y[train], y[test]
            t_train, t_test = t[train], t[test]
            v_train, v_test = v[train], v[test]

            if classic == True:
                model.fit(decay_norm(x=v_train, t_stamps = t_train, decay = decay), y_train)
                score = model.score(decay_norm(x=v_test, t_stamps = t_test, decay = decay), y_test)
                print("%s: %.2f%%" % ("accuracy", score*100))
                cvscore.append(score*100) 
            elif trainable == True:
                model.fit(x_train, y_train, batch_size = batch_size, nb_epoch = nb_epoch, verbose = 1)
                score = model.evaluate(x_test, y_test, batch_size = batch_size, verbose = 1)
                print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
                cvscore.append(score[1]*100)            
            else:
                model.fit(x = [v_train, decay_factors], y = y_train, batch_size = batch_size, nb_epoch = nb_epoch, verbose = 1)
                score = model.evaluate(x_test, y_test, batch_size = batch_size, verbose = 1)
                print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
                cvscore.append(score[1]*100)            
                
        temp = {'model':option, 'decay':decay}
        temp.update(preset)
        temp.update({'mean_score': np.mean(cvscore), 'std': np.std(cvscore)})
        data.append(temp)   
    return (data)      
        
    
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
