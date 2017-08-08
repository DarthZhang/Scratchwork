# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:25:45 2017

@author: andy
"""

import sys
import pandas as pd
import numpy as np
import pickle
import math, random
import datetime, time
import tensorflow as tf
import gensim

#from pandas.tools.plotting import scatter_matrix
from scipy import stats
from time import sleep
from sklearn.preprocessing import Imputer, normalize, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle

from keras.preprocessing import sequence
#from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, merge, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.regularizers import l1, l2


x_file = ''
y_file = ''

def main(X,Y):
    
    skf = StratifiedKFold(n_splits=5, random_state = 8)
    model = lstm_model()
    
    #for splitting purposes, use y: [0-5] for groups
    Y = Y.tolist()
    y = [l.index(1) for l in Y]
    Y = np.array(Y)
    
    #standardize X
    tmp = []
    for i in X:
        for j in i:
            tmp.append(j)
    scaler = StandardScaler()
    scaler = scaler.fit(tmp)
    for i in range(len(X)):
        X[i] = scaler.transform(X[i])
    
    #format: data[count][loss/auc] [mean_tr, mean_te]
    data = {}
    
    start = time.time(); count = 0
    for train_index, test_index in skf.split(X, y):
        count +=1
        print ("KFold #{0}, TIME: {1} Hours".format(count, (time.time() - start)/3600))
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
               
        data[count] = {}
                
        for epoch in range(10):
            #X_train, y_train = shuffle(X_train, y_train, random_state =8)
            #X_test, y_test = shuffle(X_test, y_test, random_state = 8)
            y_pred = []
            tr_acc = []; mean_tr_auc = []
            tr_loss = []; mean_tr_loss = []
            for i in range(len(X_train)):
                sleep(.0001)
                y_true = y_train[i]
                timesteps = len(X_train[i])
                for j in range(timesteps):
                    loss, acc = model.train_on_batch(X_train[i][j].reshape(1, 1, X_train[i][j].shape[0]), y_true.reshape(1,5))
                    tr_acc.append(acc)                    
                    tr_loss.append(loss)
                model.reset_states()
                
                for j in range(timesteps):
                    yhat= model.predict_on_batch(X_train[i][j].reshape(1, 1, X_train[i][j].shape[0]))
                model.reset_states()
                y_pred.append(yhat.reshape(y_true.shape))
                
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%" % ('='* int((i+1)/(len(X_train)/20)), int(5*(i+1))/(len(X_train)/20)))
                sys.stdout.write(', Acc = %.3f' % (np.mean(tr_acc)))
                sys.stdout.write(', Loss = %.3f' % (np.mean(tr_loss)))
                sys.stdout.flush()
                
            sys.stdout.write('/n')
            y_pred = np.array(y_pred)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for idx in range(Y[0].shape[0]):
                fpr[idx], tpr[idx], _ = roc_curve(y_train[:, idx], y_pred[:, idx])
                roc_auc[idx] = auc(fpr[idx], tpr[idx])
            
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_train.ravel(), y_pred.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])          
        
            mean_tr_auc.append(roc_auc)
            mean_tr_loss.append(np.mean(tr_loss))
            
            #print progress            
            print('EPOCH {0}'.format(epoch))
            print('AUROC (Micro) Training =  {0}'.format(roc_auc['micro']))
            print('Time: {0}'.format((time.time() - start)/60))
            print ('++++++++++++++++++')
        
            y_pred = []
            te_acc = []; mean_te_auc = []
            te_loss = []; mean_te_loss = []
            for i in range(len(X_test)):
                y_true = y_test[i]                
                timesteps = len(X_test[i])
                for j in range(timesteps):
                    loss, acc = model.test_on_batch(X_test[i][j].reshape(1, 1, X_test[i][j].shape[0]), y_true.reshape(1,5))
                    te_acc.append(acc)                    
                    te_loss.append(loss)
                model.reset_states()
                
                for j in range(timesteps):
                    yhat = model.predict_on_batch(X_test[i][j].reshape(1, 1, X_test[i][j].shape[0]))
                model.reset_states()
                y_pred.append(yhat.reshape(y_true.shape))
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%" % ('='* int((i+1)/(len(X_test)/20)), int(5*(i+1))/(len(X_test)/20)))
                sys.stdout.write(', Acc = %.3f' % (np.mean(te_acc)))
                sys.stdout.write(', Loss = %.3f' % (np.mean(te_loss)))
                sys.stdout.flush()
            sys.stdout.write('/n')
            y_pred = np.array(y_pred)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for idx in range(Y[0].shape[0]):
                fpr[idx], tpr[idx], _ = roc_curve(y_test[:, idx], y_pred[:, idx])
                roc_auc[idx] = auc(fpr[idx], tpr[idx])
            
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])              
           
            mean_te_auc.append(roc_auc)
            mean_te_loss.append(np.mean(te_loss))
            print ('AUROC (Micro) Testing = {0}'.format(roc_auc['micro']))
            print ('Time: {0}'.format((time.time() - start)/60))
            print ('++++++++++++++++++')
        data[count]['AUC'] = [mean_tr_auc, mean_te_auc]
        data[count]['LOSS'] = [mean_tr_loss, mean_te_loss]

            
def lstm_model(dropout_W = 0.5, dropout_U = 0.2, optimizer = 'adam', neurons = 100, learn_rate = 1e-3, W_regularizer = None, U_regularizer = None, init_mode = 'glorot_uniform'):
    model = Sequential()
    model.add(LSTM(100, batch_input_shape=(1, 1, 19), return_sequences=False, stateful=True))
    #model.add(BatchNormalization())
    #model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    #model.add(LSTM(100, batch_input_shape=(1, 1, 19), return_sequences = False, stateful = True))    
    #model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    
    model.add(Dense(5, init = 'glorot_uniform'))
    #model.add(BatchNormalization())
    model.add(Activation('softmax'))
    
    optimizer = Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return (model)

if __name__ == '__main__':
    X = np.load(x_file)
    Y = np.load(y_file)
    
    main(X, Y)