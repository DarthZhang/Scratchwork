import sys
import pickle
import os.path as path

import csv
import gzip
#import MySQLdb as mysql
import pymysql as mysql
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
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
from itertools import combinations, tee, chain
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

##### Import Dependencies ######
################################

import DatabaseBuilder as DB
import Disease
from Discretize import Lab
import PATIENT as Patient


def main():
    
    flags = make_database(conn)
    lib = discretize(conn, flags)
    #CHF, Afib, STROKE, SEPSIS, RF, CIRRHOSIS, T2DM, CAD, ATH, ARDS = querying(conn)
    #dz = [CHF, Afib, STROKE, SEPSIS, RF, CIRRHOSIS, T2DM, CAD, ATH, ARDS]
    dz = querying (conn)
    sentences = embedding(conn, lib)
    modeling(conn, sentences, lib, dz)

def make_database(conn):
    DB.make_sql(conn, admissions_doc, diagnoses_doc, icds_doc, procedures_doc, labevents_doc, items_doc, labitems_doc, patients_doc)
    DB.make_ufm(conn, engine)
    DB.make_demo(conn, engine)
    flags = DB.filter_labs(conn)
    return (flags)

def discretize(conn, flags):
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS Discrete_Labs')
    
    sql = "SELECT DISTINCT FEATURE from UFM where TYPE = '%s'" % ('l')
    #sql = "SELECT DISTINCT FEATURE from UFM"
    lst = pd.read_sql(sql = sql, con = conn)
    lst = list(lst['FEATURE'])
    
    count = 0
    l_values = []
    
    for i in lst:
        sql = "SELECT * from UFM where FEATURE = '%s'" % (i)
        df= pd.read_sql(sql = sql, con = conn)
        lab = Lab(df, flags)
        lab.is_discrete()
        lab.clean_list()
        l_values.append([i,lab.max])
        count+=1
        #Make the dataframe into SQL table:
        transfer.to_sql(lab.df, name = 'D_UFM', con=engine, index=False, index_label = 'ROW_ID', if_exists = 'append')
        print ("++++++++++++++++++++++")
        print ("Session: {0}".format(count))
        print (lab.df.head())
    
    df2= pd.read_sql("SELECT * from UFM where TYPE = 'p' or TYPE = 'd'", conn)
    df2['DISCRETE_VALUE'] = 1
    transfer.to_sql(df2, name = 'D_UFM', con=engine, index=False, index_label = 'ROW_ID', if_exists = 'append')

    n_values = [[x,1] for x in list(set(df2.FEATURE))]
    
    print ("UFM Table is Discretized.")
    
    values = l_values + n_values
    lib =[]; count = 0
    for v in values:
        if np.isnan(v[1]) or v[1]<1:
            pass
        elif v[1] == 1: 
            lib.append((count, v[0]+'_'+'1', 0))
            count+=1
        else:
            for i in range(0, v[1]+1): 
                lib.append((count, v[0] + '_' + str(i), 0))
                count+=1
    c.close()
    return (lib)

    
def querying(conn):
    chf = ['40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493', '4280', '428', '4281', '42820', '42822', '42830', '42832', '42840', '42842', '4289']
    afib = ['42731', '4271', '42789']
    #lmyc = ['1629']
    #pten = ['1830', '193', '2330', '1930']
    stroke = ['43491', '43411', '4349', '43401', '434', '4340', '43401', '43410','43490'] 
    sepsis = ['99591', '99592', '0389', '0380','0381', '03811', '03812', '03819', '03810', '0382', '0383', '0384', '0388', '03840', '03841', '03842', '03843', '03844', '03849']
    rf = ['5845','5849','5856', '5846', '5847', '5848', '5851', '5852', '5853', '5854', '5855', '5859']
    cirrhosis = ['5712','5715']
    t2dm = ['2500', '25000', '25001', '25002', '25003', '2501', '25010', '25011', '25012','25013', '2502', '25020', '25021', '25022', '25023', '25030', '2503', '25031', '25032', '25033', '25040', '25041', '25042', '25043', '2504', '2505', '25050', '25051', '25052', '25053', '2506', '25060', '25061', '25062', '25063', '2507', '25070', '25071', '25072', '25073', '2508', '25080', '25081', '25082', '25083', '2509', '25090', '25091', '25092', '25093']
    cad = ['414', '4140','41400', '41401', '41402', '41403','41404','41405','41406','41407', '4141', '41410', '41411', '41412', '41419', '4142', '4143', '4144', '4148', '4149']
    ath = ['4400', '44000', '4401', '44010', '4402', '44020', '4403', '44030', '4404', '44040','4408', '44080', '4409', '44090', '44021', '44022', '44023', '44024', '44029', '44031', '44032']
    ards = ['51881', '51884', '51883'] #EXCLUDING trauma/surgery related resp. failure.

    CHF = Disease(chf, conn)
    Afib = Disease(afib, conn)
    #LMYC = Disease(lmyc, conn)
    #PTEN = Disease(pten, conn)
    STROKE = Disease(stroke, conn)
    SEPSIS = Disease(sepsis, conn)
    RF = Disease(rf, conn)
    CIRRHOSIS = Disease(cirrhosis, conn)
    T2DM = Disease(t2dm, conn)
    CAD = Disease(cad, conn)
    ATH = Disease(ath, conn)
    ARDS = Disease(ards, conn)
    
    
    CHF.readmission(); print ("{0} done.".format(CHF))
    Afib.readmission(); print ("{0} done.".format(Afib))
    #LMYC.readmission()
    #PTEN.readmission()
    STROKE.readmission(); print ("{0} done.".format(STROKE))
    SEPSIS.readmission(); print ("{0} done.".format(SEPSIS))
    RF.readmission(); print ("{0} done.".format(RF))
    CIRRHOSIS.readmission(); print ("{0} done.".format(CIRRHOSIS))
    T2DM.readmission(); print ("{0} done.".format(T2DM))
    CAD.readmission(); print ("{0} done.".format(CAD))
    ATH.readmission(); print ("{0} done.".format(ATH))
    ARDS.readmission(); print ("{0} done.".format(ARDS))
    
    d=[]
    d.append((ATH.pos, ATH.neg))
    d.append((T2DM.pos, T2DM.neg))
    d.append((Afib.pos, Afib.neg)); d.append((ARDS.pos, ARDS.neg)); d.append((CAD.pos, CAD.neg)); d.append((CHF.pos, CHF.neg)); d.append((STROKE.pos, STROKE.neg)); d.append((CIRRHOSIS.pos, CIRRHOSIS.neg)); d.append((RF.pos, RF.neg)); d.append((SEPSIS.pos, SEPSIS.neg))
    #return (CHF, Afib, STROKE, SEPSIS, RF, CIRRHOSIS, T2DM, CAD, ATH, ARDS)
    return (d)

def embedding(conn, lib):
    pts = pd.read_sql("SELECT DISTINCT SUBJECT_ID from UFM", conn)
    pts =list(set(pts.SUBJECT_ID))

    #even faster?
    keys = [k[1] for k in lib]
    count =0; sentences = []
    c = conn.cursor()
    for p in pts:
        sql = "SELECT FEATURE, DISCRETE_VALUE, HADM_ID, TIME from UFM2  WHERE SUBJECT_ID = '%s' ORDER BY HADM_ID, TIME" %(p)
        print ("Sess: {0}".format(count))
        c.execute(sql)
        lst = list(c.fetchall())
        hadm = list(set([h[2] for h in lst]))
        for h in hadm:
            temp = sorted([i for i in lst if i[2] == h], key = lambda x: x[3])
            sentence = [i[0]+'_'+str(i[1]) for i in temp if (i[0]+'_'+str(i[1])) in keys]
            t = [pd.to_datetime(i[3]) for i in temp if (i[0] + '_' + str(i[1])) in keys]
            sentences.append((p, h, sentence, t))
        count+=1
        
    c.close()
    #skip-gram model
    #SG = gensim.models.Word2Vec(sentences = sentences, sg = 1, size = 300, window = 10, min_count = 465, hs = 1, negative = 0, workers = 4)
    
    #CBOW model
    #CBOW = gensim.models.Word2Vec(sentences = sentences, sg = 0, size = 300, window = 10, min_count = 465, hs = 1, negative = 0, workers = 4)

    #memmap implementation
    #sentences = np.array(sentences)
    #sentences2 = np.memmap(file_a, mode = 'w+', shape = (sentences.shape[0], sentences.shape[1]), dtype = 'object')
    #sentences2[:] = sentences

    #HDF5 store
    #df = pd.DataFrame(sentences, columns = ['SUBJECT_ID','HADM_ID','WORDS','TIME'])
    #store = pd.HDFStore('/home/andy/Desktop/MIMIC/sentences/sentences.h5')
    #for p in pts:
    #    store[str(p)] = df[df['SUBJECT_ID']==p]
    
    return (sentences)

##### Under Construction #####  
    
def d_cnn_train(input_shape, dropout_W = 0.2, dropout_U = 0.2):
    model = Sequential()
    model.add(Convolution1D(input_shape = input_shape, nb_filter = 300, filter_length = 3, border_mode = 'same', activation = 'relu'))
    model.add(MaxPooling1D(pool_length = 2))
    model.add(LSTM(100, dropout_W = dropout_W, dropout_U = dropout_U))
    #model.add(Dense(50, activation = 'relu'))
    #model.add(Dense(25, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return (model)

def d_lstm_train(input_shape, dropout_W = 0.2, dropout_U = 0.2):
    model = Sequential()
    model.add(LSTM(100, input_shape = input_shape, dropout_W = dropout_W, dropout_U = dropout_U))
    #model.add(Dense(50, activation = 'relu'))
    #model.add(Dense(25, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return (model)
    
def cnn_train(top_words, max_length, embedding_length, dropout_W = 0.2, dropout_U = 0.2, optimizer = 'Adam', neurons = 100, learn_rate = .01, momentum= 0.0, W_regularizer = None, U_regularizer = None, init_mode = 'zero'):
    model = Sequential()
    model.add(Embedding(top_words, embedding_length, input_length=max_length, init = init_mode))    
    model.add(Convolution1D(nb_filter = 300, filter_length = 3, border_mode = 'same', activation = 'relu'))
    model.add(MaxPooling1D(pool_length = 2))
    model.add(LSTM(output_dim=neurons, dropout_W = dropout_W, dropout_U = dropout_U, W_regularizer = W_regularizer, U_regularizer = U_regularizer))
    #model.add(Dense(50, activation = 'relu'))
    #model.add(Dense(25, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    if optimizer == 'SGD':
        optimizer = SGD(lr = learn_rate, momentum = momentum)
    elif optimizer == 'RMSprop':
        optimizer = RMSprop(lr = learn_rate)
    elif optimizer == 'Adam':
        optimizer = Adam(lr=learn_rate)
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return (model)
    
def lstm_train(top_words, max_length, embedding_length, dropout_W = 0.2, dropout_U = 0.2, optimizer = 'adam', neurons = 100, learn_rate = .01, momentum= 0.0, W_regularizer = None, U_regularizer = None, init_mode = 'zero'):
    #top_words = 9444
    #embedding_length = 300
    #max_length = 1000
    model = Sequential()
    model.add(Embedding(top_words, embedding_length, input_length=max_length))    
    model.add(LSTM(output_dim=neurons, dropout_W = dropout_W, dropout_U = dropout_U, W_regularizer = W_regularizer, U_regularizer = U_regularizer))
    #model.add(Dense(50, activation = 'relu'))
    #model.add(Dense(25, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    if optimizer == 'SGD':
        optimizer = SGD(lr = learn_rate, momentum = momentum)
    elif optimizer == 'RMSprop':
        optimizer = RMSprop(lr = learn_rate)
    elif optimizer == 'Adam':
        optimizer = Adam(lr=learn_rate)
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return (model)
    
def modeling(conn, sentences, lib, dz):
#def modeling(conn, df, lib, dz):
  
    #pts = pd.read_sql("SELECT DISTINCT SUBJECT_ID from UFM", conn)
    #pts =list(set(pts.SUBJECT_ID))
    #pool = []
    #for d in dz:
    #    pool += d.pos + d.neg
    np.random.seed(7)
    decay = .0002
    data = []; train = []; test = []
    keys = [k[1] for k in lib]
    
    admits = pd.read_sql("SELECT * from admissions", conn)
    
    for itr in range(0,5):
        print ("Sess: {0}".format(itr))
        for d in dz:
            neg = random.sample(d[1], len(d[0]))
            temp = d[0] + neg
            random.shuffle(temp)
            t1, t2 = cross_validation.train_test_split(temp, test_size = .2)
            train +=t1; test +=t2
                    
        #X stands for raw indexes of feature input; V stands for raw feature input
        #W stands for word vectors from feature input trained by Word2Vec
        X_train = []; t_train = []; W_train = []; Y_train = []
        X_test = []; t_test = []; W_test = []; Y_test = []
        V_train = []; V_test = []
    
        count=0
        for t in train:
            print (count)
            count+=1

            corpus = [[s[2], s[3]] for s in sentences if  (s[0] == t[0]) and (pd.to_datetime(admits[admits['HADM_ID']==s[1]].ADMITTIME.values[0]) <= t[1])]
            #order subject by time of entry for each sentence (admission)
            corpus = sorted(corpus, key = lambda x: x[1])
            #transpose into nx2xd from 2xnxd
            #this way, corpus[0] refers to words and corpus[1] refers to times
            corpus = list(map(list, zip(*corpus)))                  
            x_train = list(chain.from_iterable(corpus[0]))
            t_stamps = list(chain.from_iterable(corpus[1]))
            x = np.array(list(map(lambda x: keys.index(x), x_train)))
     
            #configure each timestamp to reflect time elapsed from first time entry
            #calculate time decay from initial event
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
            
####### DECAY STEP ##############      
        #make exclusion list for test patients
        introns = [t[0] for t in test]
        #instance = [t[2] for t in test]
        #sentences have format (s, hadm, [words], [times])
        lst = [i[2] for i in sentences if i[0] not in introns]
            
        #lst = list(~df[df.SUBJECT_ID.isin(introns)].SUBJECT_ID)
        #word2vec:
        #configure hyperparams as appropriate
        print ("Making SG...")
        SG = gensim.models.Word2Vec(sentences = lst, sg = 1, size = embedding_length, window = 10, min_count = 50, hs = 1, negative = 0, workers = 4)
        print ("SG embedding complete.")
        #CBOW = gensim.models.Word2Vec(sentences = lst, sg = 0, size = 300, window = 10, min_count = 465, hs = 1, negative = 0, workers = 4)
        #construct sequence feature from train(ing) set
        #making W_train, W_test
        cnn_d = d_cnn_train(input_shape = (max_review_length, embedding_length))
        lstm_d = d_lstm_train(input_shape = (max_review_length, embedding_length))

        #SG = gensim.models.Word2Vec.load('/home/andy/Desktop/MIMIC/sg_temp')
        nb_epoch = 5
        for e in range(nb_epoch):
            print ("+++++++++++++")
            print ("Epoch: %d" %e)
            
            for i in range(0,len(V_train), 128):
                if (i+128)>=len(t_train): 
                    t_stamps = t_train[i:]
                    x = V_train[i:]
                    y = Y_train[i:]
                else:
                    t_stamps = t_train[i:i+128]
                    x = V_train[i:i+128]
                    y = Y_train[i:i+128]

                W_train = []
                for ii in range(len(t_stamps)):
                    decay_factor=np.array([math.exp(-1 * decay * elapse.total_seconds()/3600) for elapse in t_stamps[ii]])
                    v = np.array(list(map(lambda x: SG[x] if x in SG.wv.vocab else [0]*embedding_length, x[ii])))
                    w = np.array([np.multiply(v[index], decay_factor[index]) for index in range(len(decay_factor))])
                    if w.shape[0]<max_review_length:
                        temp = np.array([[0]*embedding_length for jjj in range(max_review_length-len(v))])
                        w = np.concatenate((w,temp))
                    elif len(w) > max_review_length:
                        w = w[0:max_review_length]
                    W_train.append(w)
                    
                W_train = np.array(W_train)
                lstm_d.fit(W_train, y, validation_split = .2,  nb_epoch = 5, verbose = 1)
                cnn_d.fit(W_train, y, validation_split = .2,  nb_epoch = 5, verbose= 1)
##########################################
                
        #training normal LSTM and CNN-LSTM          
        top_words = [9444]
        max_review_length = [1000]
        embedding_length = [300]          
        X_train = sequence.pad_sequences(X_train, maxlen=max_review_length[0])
        X_test = sequence.pad_sequences(X_test, maxlen=max_review_length[0])


        #build model using KerasClassifier and Gridsearch
        cnn = KerasClassifier(build_fn=cnn_train, verbose=1)
        lstm = KerasClassifier(build_fn=lstm_train, verbose=1)
        # define the grid search parameters

        batch_size = [32, 64, 128]
        epochs = [20, 50, 100]
        optimizer = ['SGD', 'RMSprop', 'Adam']
        learn_rate = [0.00001, 0.0001, 0.001]
        momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
        neurons = [50, 100, 200]
        dropout_W = [.1, .2, .5]
        dropout_U = [.1, .2, .5]
        W_regularizer = [l1(.0001), l1(.001), l1(.01), l2(.0001), l2(.001), l2(.01), None]
        U_regularizer = [l1(.0001), l1(.001), l1(.01), l2(.0001), l2(.001), l2(.01), None]
        init_mode = ['uniform', 'normal', 'zero']
        #activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
        param_grid = dict(top_words=top_words, max_length = max_review_length, embedding_length = embedding_length, batch_size=batch_size, nb_epoch=epochs, optimizer = optimizer, learn_rate = learn_rate, momentum = momentum, neurons = neurons, dropout_W = dropout_W, dropout_U = dropout_U, W_regularizer = W_regularizer, U_regularizer = U_regularizer, init_mode = init_mode)
            
        #setup GridSearch w/ cross validation
        cnn_grid = GridSearchCV(estimator=cnn, param_grid=param_grid, scoring = 'roc_auc', cv = 5, n_jobs=-1)
        lstm_grid = GridSearchCV(estimator=lstm, param_grid=param_grid, scoring = 'roc_auc', cv = 5, n_jobs=-1)
        # Fit the model
        cnn_result = cnn_grid.fit(X_train, Y_train)
        lstm_result = lstm_grid.fit(X_train, Y_train) 
        #grid_search results:
        print("CNN Best: %f using %s" % (cnn_result.best_score_, cnn_result.best_params_))
        means = cnn_result.cv_results_['mean_test_score']
        stds = cnn_result.cv_results_['std_test_score']
        params = cnn_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, params))
        
        print("LSTM Best: %f using %s" % (cnn_result.best_score_, cnn_result.best_params_))
        means = lstm_result.cv_results_['mean_test_score']
        stds = lstm_result.cv_results_['std_test_score']
        params = lstm_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, params))
        
        #KFold = 5
        #kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
        #cvscores = []
        #for training, testing in kfold.split(X_train, Y_train):     
            # Fit the model
            #model.fit(X[training], Y[training], nb_epoch=150, batch_size=10, verbose=0)
            # evaluate the model
            #scores = model.evaluate(X[testing], Y[testing], verbose=0)
            #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            #cvscores.append(scores[1] * 100)
        #print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

 ######TESTING#######
        cnn = cnn_train(top_words = top_words, max_length = max_review_length, embedding_length=embedding_length)
        lstm = lstm_train(top_words = top_words, max_length = max_review_length, embedding_length=embedding_length)
            
        cnn.fit(X_train, Y_train, validation_split = .2, nb_epoch=100, batch_size=128, shuffle = True, verbose=1)
        lstm.fit(X_train, Y_train, validation_split = .2, nb_epoch=100, batch_size=128, shuffle = True, verbose=1)

        #testing
        predictions_lstm = lstm.predict_classes(X_test)
        predictions_cnn = cnn.predict_classes(X_test)

        acc = accuracy_score(Y_test, predictions_lstm)
        f1 = f1_score (Y_test, predictions_lstm)
        auc = roc_auc_score (Y_test, predictions_lstm)
        scores_lstm = [("Accuracy", acc) , ("F1 Score", f1) , ("AUC Score",auc)]

        acc = accuracy_score(Y_test, predictions_cnn)
        f1 = f1_score (Y_test, predictions_cnn)
        auc = roc_auc_score (Y_test, predictions_cnn)
        scores_cnn = [("Accuracy", acc) , ("F1 Score", f1) , ("AUC Score",auc)]

        print ("LSTM DATA: ")
        for s in scores_lstm:
            print("%s: %.2f" %(s[0], s[1]), end = " ")
        print ("")
        print ("CNN DATA: ")
        for s in scores_cnn:
            print("%s: %.2f" %(s[0], s[1]), end = " ")        
        
        
        data.append(data)
            
    return (Data)

##############################

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    desc = "Welcome to PipeLiner by af1tang."
    version = "version 1.0"
    opt = OptionParser (description = desc, version=version)
    opt.add_option ('-i', action = 'store', type ='string', dest='input', help='Please input path to Database File.')
    opt.add_option ('-o', action = 'store', type = 'string', dest='output', default='CHF_data.pickle', help='Please state desired storage file for this session.')
    (cli, args) = opt.parse_args()
    opt.print_help()
    
    mimic = 'MIMIC3'
    host = 'illidan-gpu-1.egr.msu.edu'
    user = 'af1tang'
    pw = 'illidan'    
    port = 3306
    
    #connect to MySQL using engine to write pandas df --> mysql
    conn = mysql.connect(host = host, user = user, passwd = pw, db = mimic, port = port)    
    engine = create_engine ("mysql+pymysql://af1tang:illidan@illidan-gpu-1.egr.msu.edu:3306/MIMIC3")
    
    main()  
    
    conn.close()
