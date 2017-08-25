# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:20:56 2017

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
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC as SVM
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, f1_score
from sklearn.utils import shuffle, class_weight


x_file = ''
y_file = ''

def main(x, y, task):
    
    #ys = [yr, ym, y25]
    #y_names = ['readm', 'mort_h', 'pheno25']
    #xs = [x48, onehot, w2v, w48, sentences]
    #x_names = ['48h', 'sparse_dx', 'w2v', 'w2v_48h', 'sentences']
    
    lr = LR(C = 1e-4, penalty = 'l2', verbose = 1)    #sag if multiclass/multilabel
    svm = SVM(C=1e5, verbose = True)
    rf = RF(n_estimators = 60, verbose = 1)
    gbc = GBC(n_estimators = 200, learning_rate = 1e-3, verbose = 1)
    
    models = [lr, svm, rf, gbc]
    names = ['LR', 'SVM', 'RF', 'GBC']
    data = {}    
    for idx in range(len(models)):
        if task != 'binary':
            data[names[idx]] = {}
            for ix in range(25):
                dat = run_experiment(x, y[:, ix], models[idx], task)
                data[names[idx]][ix] = dat
        else:
            dat = run_experiment(x, y, models[idx], task)
            data[names[idx]] = dat
        
    return (data)

def build_XY (x, y, option):
    
    #flatten X
    x = X_48hr (x)
    
    #w2v conversion
    w2v = W2V(dx)
    #standardize Word Vectors (optional)
    scaler = StandardScaler()
    w2v = scaler.fit_transform(w2v)
    #concatenate X with w2v
    #first, extend w2v to 3D tensor
    #w2v = np.repeat(w2v[:, np.newaxis, :], maxlen, axis=1)
    #make new X
    #x = np.concatenate((x, w2v), axis = 2)
    
    #mortality and readmission labels
    ym = y[:, 2]
    yr = y[:, -1]
    return (w2v, y)

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

def X_48hr (X):
    #standardize X
    tmp = []
    maxlen = []
    for i in range(len(X)):
        if len(X[i]) < 24: 
            pass
        else:
            tmp.append(np.array(X[i][-24:]))
        maxlen.append(len(X[i]))

    tmp = np.array(tmp).T
    tmp = tmp.reshape(tmp.shape[0], tmp.shape[1] * tmp.shape[2]).T
    scaler = StandardScaler()
    scaler = scaler.fit(tmp)
    for i in range(len(X)):
        X[i] = scaler.transform(X[i])
    del tmp
    
    x = []
    for i in X:
        mean = np.mean(i, axis = 0)
        mins = np.amin(i, axis = 0)
        maxs = np.amax(i, axis = 0)
        stds = np.std(i, axis = 0)
        skews = skew(i, axis=0)
        samples = len(i)
        vec = np.concatenate([mins, mean, maxs, stds, skews])
        vec = np.append(vec, samples)
        x.append(vec)
    x = np.array(x)
    
    return (x)
    
def run_experiment(x, y, model, task):
    skf = StratifiedKFold(n_splits=5, random_state = 8)
    data = {}
            
    start = time.time(); count = 0
    for train_index, test_index in skf.split(x, y):
        count +=1
        print ("KFold #{0}, TIME: {1} Hours".format(count, (time.time() - start)/3600))
        
        data[count] = {}
        data[count]['tr_auc'] = []
        data[count]['f1_score'] = []
        data[count]['te_auc'] = []
        data[count]['te_matrix'] = []
        
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if task == 'binary':
            xs, ys = balanced_subsample(X_train, y_train, 1.0)
            ys = np.array([[i] for i in ys])
        else:
            xs, ys = shuffle(X_train, y_train, random_state =8)
        
        model.fit(xs, ys)
        y_pred = model.predict(xs)
        yhat = model.predict(X_test)
        
        fpr = dict()
        tpr = dict()
        tr_roc_auc = dict()
        f1= dict()
        te_roc_auc = dict()
        te_matrix = dict()
        
        fpr, tpr, _ = roc_curve(ys, y_pred)
        tr_roc_auc = auc(fpr, tpr)
        te_matrix = confusion_matrix(y_test, np.array([round(i) for i in yhat]))
        f1 = f1_score(y_test, np.array([round(i) for i in yhat]))
        
        fpr, tpr, _ = roc_curve(y_test, yhat)
        te_roc_auc = auc(fpr, tpr)
        
        data[count]['tr_auc'].append(tr_roc_auc)
        data[count]['f1_score'].append(f1)
        data[count]['te_auc'].append(te_roc_auc)
        data[count]['te_matrix'].append(te_matrix)
    
    return (data)
    
def r_search(x, y):
    #random search params
    lr_params = {'penalty': ['l1', 'l2'], 'C': sp_rand(1e-5, .1)}
    svm_params = {'kernel': ['rbf', 'linear'], 'C':sp_rand (10, 1e5)}
    rf_params = {'criterion': ['gini', 'entropy'], 'n_estimators': sp_randint(50, 200), 'bootstrap': [True, False] }
    gbc_params = {'learning_rate': sp_rand(1e-6, 1e-1), 'n_estimators': sp_randint(50, 200), 'loss': ['deviance', 'exponential']}
    
    data = {}
    xs, ys = balanced_subsample(x, y)
    lst = [LR(verbose = 1), RF(verbose = 1), SVM(verbose = True), GBC(verbose = 1)]
    names = ['LR', 'RF', 'SVM', 'GB']
    params = [lr_params, rf_params, svm_params, gbc_params]
    for idx in range(len(lst)):
        n_iter_search = 60
        start = time.time()    
        rsearch = random_search(estimator = lst[idx], param_distributions = params[idx], n_iter=n_iter_search, scoring='roc_auc', fit_params=None, n_jobs=1, iid=True, refit=True, cv=5, verbose=0, random_state=8)
        rsearch.fit(xs, ys)
        data[names[idx]] = rsearch.cv_results_
        print (names[idx]+" results complete.")
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
        " parameter settings." % ((time.time() - start), n_iter_search))
    return (data)
            
    
def balanced_subsample(x,y,subsample_size=1.0):

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
    
def discretize (X, qc):
    sentences = {}
    for h in list(X.keys()):
        sentences[h] = []
    for timestep in X[h]:
        for idx in range(len(timestep)):
            q = qc[idx]
            if timestep[idx] <= q[1]: string = '_1'
            elif q[1] < timestep[idx] <= q[2]: string = '_2'
            elif q[2]<timestep[idx] <=q[3]: string = '_3'
            elif q[3] <timestep[idx]<=q[4]: string = '_4'
            elif timestep[idx] > q[4]: string = '_5'
            else: print(timestep[idx])
            sentences[h].append(str(idx) + string)
    return (sentences)

def onehot(dix):
    #make dx history one-hot
    onehot = []
    for i in dix:
        tmp = [0] * 942
        for j in i:
            tmp[j] +=1
        onehot.append(tmp)

    
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
    args = parser.parse_args("--x /home/andy/Desktop/MIMIC/vars/npy/seqs/Xts.npy --x /home/andy/Desktop/MIMIC/vars/npy/seqs/dix.npy --x /home/andy/Desktop/MIMIC/vars/npy/seqs/sentences.npy --x_name 19ts --x_name w2v --x_name sentences --y /home/andy/Desktop/MIMIC/vars/npy/Ys/Yr.npy --w2v False --w2v True --w2v True --maxlen None --maxlen 39 --maxlen 1000 --multi --mode lstm".split())
    #args = parser.parse_args()

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
    df = df.transpose()
    df.index.name = 'KFOLD'
    df.to_pickle(args.o)
