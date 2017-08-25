# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 11:40:31 2017

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
import matplotlib.pyplot as plt

import argparse

#from pandas.tools.plotting import scatter_matrix
from scipy import stats
from scipy.stats import uniform as sp_rand
from scipy.stats import randint as sp_randint
from scipy.stats import skew
from time import sleep
from datetime import timedelta
from sklearn.preprocessing import Imputer, normalize, StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV as random_search
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, f1_score
from sklearn.utils import shuffle, class_weight

from keras.preprocessing import sequence
#from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Activation, Dropout, TimeDistributed, Bidirectional, Masking, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras.preprocessing import sequence
from keras import metrics

def main(xs, x_names, dxs, y, w2vs, maxlens, multi, mode, epochs):
    
    #epochs = 30
    #x_names = ['19ts', 'w2v', 'sentences']
    #x_names = ['19ts_w2v', '19ts_icd9']
    #dxs = [None, None, None]
    #dxs = [w2v, onehot]
    #w2vs = [False, True, True]
    #w2v = False
    #multi = False
    #mode = 'hierarchal_lstm'
    #maxlens = [None, 39, 1000]
    #maxlen = None
    
    data  = {}   
    
    for xi in range(len(xs)):
        x_name = x_names[xi]
        x = xs[xi]
        dx = dxs[xi]
        maxlen = maxlens[xi]
        w2v = w2vs[xi]
        
        if w2v:
            x = [[i+1 for i in d] for d in x]
            x = sequence.pad_sequences(x, maxlen, value = 0)
            input_shape = (x.shape[1],)
            embed_shape = (942, 200, maxlen)
        else:
            input_shape = (x.shape[1], x.shape[2])
            if dx is not None:
                embed_shape = (dx.shape[1],)
            else:
                embed_shape = None
        
        params = {'input_shape':input_shape, 'embed_shape': embed_shape, 'multi': multi, 'w2v':w2v}
        dat = run_experiment(x, y, params, epochs, mode)
        data[x_name] = dat
        
    return (data)

def vocab_index (vocab):
    word2idx = vocab
    idx2word = dict([(v,k) for k, v in vocab.items()])
    return (word2idx, idx2word)
    
def W2V (dx):
    dx = np.ndarray.tolist(dx)
    SG = gensim.models.Word2Vec(sentences = dx, sg = 1, size = 200, window = 5, min_count = 0, hs = 1, negative = 0)
    weights = SG.wv.syn0
    vocab = dict([(k, v.index) for k, v in SG.wv.vocab.items()])
    w2i, i2w = vocab_index(vocab)
    
    #turn sentences into word vectors for each admission
    dx = [list(map(lambda i: w2i[i] if i in w2i.keys() else 0, vv)) for vv in dx]    
    #word vectors here
    w2v = [] 
    
    for sentence in dx:
        one_hot = np.zeros((len(sentence), weights.shape[0]))
        one_hot[np.arange(len(sentence)), sentence] = 1
        one_hot = np.sum(one_hot, axis= 0)
        w2v.append(np.dot(one_hot.reshape(one_hot.shape[0]), weights))
    return (w2v, SG, weights, vocab)
        
def run_experiment(x, dx, y, params, epochs = 30, mode = 'lstm'):
    skf = StratifiedKFold(n_splits=5, random_state = 8)
    data = {}
    
    if (params['multi'] == True) :
        tmp = y[:, -1]
    else:
        tmp = y
        
    start = time.time(); count = 0
    for train_index, test_index in skf.split(x, tmp):
        count +=1
        print ("KFold #{0}, TIME: {1} Hours".format(count, (time.time() - start)/3600))
        
        data[count] = {}
        data[count]['tr_auc'] = []
        data[count]['f1_score'] = []
        data[count]['te_auc'] = []
        data[count]['te_matrix'] = []
        
        X_train, X_test = x[train_index], x[test_index]
        if dx is not None:
            dx_train, dx_test = dx[train_index], dx[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        
        if (params['multi'] == False):
            if dx is not None:
                xs, ds, ys = hierarchal_subsample(X_train, dx_train, y_train, 1.0)
            else:
                xs, ys = balanced_subsample(X_train, y_train, 1.0)
            ys = np.array([[i] for i in ys])
            sample_wt = None
        else:
            if dx is not None:
                xs, ds, ys = shuffle(X_train, dx_train, y_train, random_state =8)
            else:
                xs, ys = shuffle(X_train, y_train, random_state =8)
            sample_wt = class_weight.compute_sample_weight('balanced', ys)
        
        model = None
        if mode == 'lstm':
            model = lstm_train(**params)
        elif mode == 'cnn':
            model = cnn_train(**params)
        elif mode == 'mlp':
            model = mlp_train(**params)
        elif mode == 'hierarchal_lstm':
            model = hierarchal_lstm(**params)
        elif mode == 'hierarchal_cnn':
            model = hierarchal_cnn(**params)
        else:
            print("ERROR IN MODE SELECTION.")
            return;
        
        if dx is not None:
            model.fit(x = [xs, ds], y= ys, epochs = epochs, sample_weight = sample_wt)
            y_pred = model.predict(x = [xs, ds])
            yhat = model.predict(x= [X_test, dx_test])
        else:
            model.fit(xs, ys, nb_epoch = epochs, sample_weight = sample_wt)
            y_pred = model.predict(xs)
            yhat = model.predict(X_test)
        
        fpr = dict()
        tpr = dict()
        tr_roc_auc = dict()
        f1= dict()
        te_roc_auc = dict()
        te_matrix = dict()
        
        if (params['multi'] == True):
            for idx in range(y[0].shape[0]):
                fpr[idx], tpr[idx], _ = roc_curve(ys[:, idx], y_pred[:, idx])
                tr_roc_auc[idx] = auc(fpr[idx], tpr[idx])
                
            fpr["micro"], tpr["micro"], _ = roc_curve(ys.ravel(), y_pred.ravel())                
            tr_roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])   
            
            for idx in range(y[0].shape[0]):
                fpr[idx], tpr[idx], _ = roc_curve(y_test[:, idx], yhat[:, idx])
                te_roc_auc[idx] = auc(fpr[idx], tpr[idx])
                te_matrix[idx] = confusion_matrix(y_test[:, idx], np.array([round(i) for i in yhat[:, idx]]))
                f1[idx] = f1_score(y_test[:, idx], np.array([round(i) for i in yhat[:, idx]]))
                
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), yhat.ravel())
            te_roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            f1["micro"] = f1_score(y_test.ravel(), np.array([round(i) for i in yhat.ravel()]))
        else:
            fpr, tpr, _ = roc_curve(ys, y_pred)
            tr_roc_auc = auc(fpr, tpr)
            te_matrix = confusion_matrix(y_test, np.array([round(i[0]) for i in yhat]))
            f1 = f1_score(y_test, np.array([round(i[0]) for i in yhat]))
            
            fpr, tpr, _ = roc_curve(y_test, yhat)
            te_roc_auc = auc(fpr, tpr)
        
        data[count]['tr_auc'].append(tr_roc_auc)
        data[count]['f1_score'].append(f1)
        data[count]['te_auc'].append(te_roc_auc)
        data[count]['te_matrix'].append(te_matrix)
    
    return (data)
    
def r_search(x, y, input_shape):
    #random search params
    mlp_params = {'units': [64, 128, 256, 512], 'rate': sp_rand(.2, .9)}
    lstm_params = {'units': [64, 128, 256, 512], 'rate': sp_rand(.2, .9)}
    cnn_params = {'filters': [32, 64, 128, 256, 512], 'filter_length': [2, 3, 4, 5, 6], 'pool_size': [2, 3]}
    
    data = {}
    xs, ys = balanced_subsample(x, y)
    lst = [mlp_train(input_shape), lstm_train(input_shape), cnn_train(input_shape)]
    names = ['MLP', 'LSTM', 'CNN']
    params = [mlp_params, lstm_params, cnn_params]
    for idx in range(len(lst)):
        n_iter_search = 60
        start = time.time()    
        rsearch = random_search(estimator = lst[idx], param_distributions = params[idx], n_iter=n_iter_search, scoring='roc_auc', fit_params=None, n_jobs=1, iid=True, refit=True, cv=3, verbose=10, random_state=8)
        rsearch.fit(xs, ys)
        data[names[idx]] = rsearch.cv_results_
        print (names[idx]+" results complete.")
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
        " parameter settings." % ((time.time() - start), n_iter_search))
    return (data)
            
    
def balanced_subsample(x, y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys

def hierarchal_subsample(x, dx, y,subsample_size=1.0):

    class_xs = []
    class_dxs = []
    min_elems_x = None
    min_elems_d = None

    for yi in np.unique(y):
        elems_x = x[(y == yi)]
        elems_d = dx[(y==yi)]
        class_xs.append((yi, elems_x))
        class_dxs.append((yi, elems_d))
        if min_elems_x == None or elems_x.shape[0] < min_elems_x:
            min_elems_x = elems_x.shape[0]
            min_elems_d = elems_d.shape[0]

    use_elems_x = min_elems_x
    use_elems_d = min_elems_d
    if subsample_size < 1:
        use_elems_x = int(min_elems_x*subsample_size)
        use_elems_d = int(min_elems_d*subsample_size)

    xs = []
    dxs = []
    ys = []

    for lst1, lst2 in zip(class_xs, class_dxs):
        ci = lst1[0]
        this_xs = lst1[1]
        this_dxs = lst2[1]
        
        if len(this_xs) > use_elems_x:
            this_xs, this_dxs = shuffle(this_xs, this_dxs)

        x_ = this_xs[:use_elems_x]
        d_ = this_dxs[:use_elems_d]
        y_ = np.empty(use_elems_x)
        y_.fill(ci)

        xs.append(x_)
        dxs.append(d_)
        ys.append(y_)

    xs = np.concatenate(xs)
    dxs = np.concatenate(dxs)
    ys = np.concatenate(ys)

    return xs, dxs, ys
    
def lstm_train(input_shape, embed_shape = (942, 200, 39), stateful = False, target_rep = False, w2v = False, multi = False):
    model = Sequential()
    if w2v:
        model.add(Embedding(input_dim = embed_shape[0], output_dim = embed_shape[1], input_length = embed_shape[2], mask_zero = True))
    else:
        model.add(Masking(mask_value=0., input_shape=input_shape))
        
    if (stateful == True):
        model.add(Bidirectional(LSTM(256, return_sequences = target_rep, stateful =stateful), merge_mode = 'concat', batch_input_shape = (1, input_shape[0], input_shape[1])))
    else:
        model.add(Bidirectional(LSTM(256, return_sequences = multi, stateful = stateful), merge_mode = 'concat'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    if multi:
        model.add(Bidirectional(LSTM(256, return_sequences = target_rep, stateful = stateful), merge_mode = 'concat'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
    
    if (target_rep == True):
        if multi:
            model.add(TimeDistributed(Dense(25, activation = 'sigmoid')))
        else:
            model.add(TimeDistributed(Dense(1, activation = 'sigmoid')))
    else:
        if multi:
            model.add(Dense(25, activation = 'sigmoid'))
        else:
            model.add(Dense(1, activation = 'sigmoid'))
    optimizer = Adam(lr=1e-4, beta_1 =.5 )
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return (model)
    
def cnn_train(input_shape, embed_shape = (942, 200, 1000), stateful = False, target_rep = False, w2v = False, multi = False):
    model = Sequential()
    if w2v:
        model.add(Embedding(input_dim = embed_shape[0], output_dim = embed_shape[1], input_length = embed_shape[2], mask_zero = False))
    model.add(Convolution1D(nb_filter = 64, filter_length = 3, border_mode = 'same', activation = 'relu'))
    model.add(MaxPooling1D(pool_length = 3))
    
    model.add(Bidirectional(LSTM(256, return_sequences = target_rep, stateful = stateful), merge_mode = 'concat'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    
    #model.add(Bidirectional(LSTM(256, return_sequences = target_rep, stateful = stateful), merge_mode = 'concat'))
    #model.add(Activation('tanh'))
    #model.add(Dropout(0.5))
    
    if (target_rep == True):
        if multi:
            model.add(TimeDistributed(Dense(25, activation = 'sigmoid')))
        else:
            model.add(TimeDistributed(Dense(1, activation = 'sigmoid')))
    else:
        if multi:
            model.add(Dense(25, activation = 'sigmoid'))
        else:
            model.add(Dense(1, activation = 'sigmoid'))
    optimizer = Adam(lr=1e-4, beta_1 =.5 )
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return (model)

def mlp_train(input_shape, multi = False):
    model = Sequential()
    model.add(Dense(512, activation = 'relu', input_shape = input_shape))
    model.add(Dropout(.5))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(.5))

    if multi:
        model.add(Dense(25, activation = 'sigmoid'))
    else:
        model.add(Dense(1, activation = 'sigmoid'))
    optimizer = Adam(lr=1e-4, beta_1 =.5 )
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return (model)

def hierarchal_cnn (input_shape, embed_shape, stateful= False, target_rep = False, multi = False, w2v= False):
    x = Input(shape = input_shape, name = 'x')
    
    xx = Convolution1D(nb_filter = 64, filter_length = 3, border_mode = 'same', activation = 'relu') (x)
    xx = MaxPooling1D(pool_length = 3) (xx)
    
    xx = Bidirectional(LSTM (256, return_sequences = False, stateful = stateful, activation = 'relu'), merge_mode = 'concat') (xx)
    xx = Dropout(0.5)(xx)
    
    dx = Input(shape = embed_shape, name = 'dx')

    xx = concatenate([xx, dx])
    #xx = Dense(256, activation = 'relu') (xx)
    if multi:
        y = Dense(25, activation = 'sigmoid') (xx)
    else:
        y = Dense(1, activation = 'sigmoid') (xx)
    model = Model(inputs = [x, dx], outputs = [y])
    model.compile (loss = 'binary_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['accuracy'])
    return (model)

def hierarchal_lstm (input_shape, embed_shape, stateful= False, target_rep = False, multi = False, w2v = False):
    x = Input(shape = input_shape, name = 'x')
    xx = Masking(mask_value=0.)(x)
    
    xx = Bidirectional(LSTM (256, return_sequences = multi, stateful = stateful, activation = 'relu'), merge_mode = 'concat') (xx)
    xx = Dropout(0.5)(xx)
    if multi:
        xx = Bidirectional(LSTM (256, return_sequences = target_rep, stateful = stateful, activation = 'relu'), merge_mode = 'concat') (xx)
        xx = Dropout(0.5)(xx)
    
    dx = Input(shape = embed_shape, name = 'dx')

    xx = concatenate([xx, dx])
    xx = Dense(256, activation = 'relu') (xx)
    if multi:
        y = Dense(25, activation = 'sigmoid') (xx)
    else:
        y = Dense(1, activation = 'sigmoid') (xx)
    model = Model(inputs = [x, dx], outputs = [y])
    model.compile (loss = 'binary_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['accuracy'])
    return (model)

def str2bool(v):
    return v.lower() in ("True", "true", "yes", "Yes", 't', 'T', '1', 'YES', 'Y', 'y')
    
if __name__ == '__main__':
    xs = []; dxs = []; maxlens = []
    
    parser = argparse.ArgumentParser("Input X, Y, and settings.")
    parser.add_argument('--x', type = str, action = 'append', help = "Enter X's.")
    parser.add_argument('--x_name', type = str, action = 'append', help = "Enter X names.")
    parser.add_argument('--dx', action = 'append', default = None, help = "Enter Auxiliary Inputs.")
    parser.add_argument('--y', type = str, help = 'Enter task w/ labels Y.')
    parser.add_argument('--w2v', type = str2bool, action = 'append', help = "Do you want word2vec embeddings on input?")
    parser.add_argument('--maxlen', default =None, action = 'append', help = "Enter Padding length.")
    parser.add_argument('--multi', action = 'store_true', help = "Multilabel classification?")
    parser.add_argument('--mode', choices = ['lstm', 'hierarchal_lstm', 'cnn', 'hierarchal_cnn'])
    parser.add_argument('--epochs', type = int, default = 30, help = "Enter number of epochs to train model on.")
    parser.add_argument('--o', help = "Enter Output File name.")
    #args = parser.parse_args("--x /home/andy/Desktop/MIMIC/vars/npy/seqs/Xts.npy --x /home/andy/Desktop/MIMIC/vars/npy/seqs/dix.npy --x /home/andy/Desktop/MIMIC/vars/npy/seqs/sentences.npy --x_name 19ts --x_name w2v --x_name sentences --y /home/andy/Desktop/MIMIC/vars/npy/Ys/Yr.npy --w2v False --w2v True --w2v True --maxlen None --maxlen 39 --maxlen 1000 --multi --mode lstm --o /home/andy/Desktop/MIMIC/dat/LSTM results/readm.pkl".split())
    args = parser.parse_args()

    for x_file in args.x:
        x = np.load(x_file)
        xs.append(x)
        
    try:
        for dx_file in args.dx:
            dx = np.load(dx_file)
            dxs.append(dx)
    except: dxs = None
    
    for m in args.maxlen:
        try: maxlens.append(int(m))
        except: maxlens.append(None)
        
    y = np.load(args.y)    
    
    data = main(xs = xs, x_names = args.x_name, dxs = dxs, y= y, w2vs = args.w2v, maxlens = maxlens, multi = args.multi, mode = args.mode, epochs = args.epochs)
    
    df = pd.DataFrame(data)
    df.index.name = 'KFOLD'
    df.to_pickle(args.o)

