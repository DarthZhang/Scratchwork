# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:38:42 2017

@author: andy
"""

import sys
import _pickle as pickle
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
from itertools import combinations
from datetime import date
from datetime import time
from datetime import timedelta
from sklearn import preprocessing

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding


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
    CHF, Afib, STROKE, SEPSIS, RF, CIRRHOSIS, T2DM, CAD, ATH, ARDS = querying(conn)
    dz = [CHF, Afib, STROKE, SEPSIS, RF, CIRRHOSIS, T2DM, CAD, ATH, ARDS]
    sentences = embedding(conn, lib)
    modeling(conn, sentences, dz)

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
    return (CHF, Afib, STROKE, SEPSIS, RF, CIRRHOSIS, T2DM, CAD, ATH, ARDS)
    

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

    return (sentences)

##### Under Construction #####  
def training (X, W, Y):
    np.random.seed(100)
    models = []
    top_words = 9444

    #CNN with decay
    model = Sequential()
    model.add(Convolution1D(nb_filter = 300, filter_length = 3, border_mode = 'same', activation = 'relu'))
    model.add(MaxPooling1D(pool_length = 2))
    model.add(LSTM(100))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    print(model.summary())
    model.fit(W, Y, nb_epoch = 10, batch_size = 1000)
    models.append(("Decay CNN", model))
    
    #Non-decay CNN
    x = sequence.pad_sequences(X)
    embedding_length = 300
    model = Sequential()
    model.add(Embedding(top_words, embedding_length))    
    model = Sequential()
    model.add(Convolution1D(nb_filter = 300, filter_length = 3, border_mode = 'same', activation = 'relu'))
    model.add(MaxPooling1D(pool_length = 2))
    model.add(LSTM(100))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    print(model.summary())
    model.fit(x, Y, nb_epoch = 10, batch_size = 1000)
    models.append(("Non-decay CNN", model))
    
    #Non-decay CNN with dropouts
    x = sequence.pad_sequences(X)
    embedding_length = 300
    model = Sequential()
    model.add(Embedding(top_words, embedding_length, dropout=0.2))    
    model = Sequential()
    model.add(Convolution1D(nb_filter = 300, filter_length = 3, border_mode = 'same', activation = 'relu'))
    model.add(MaxPooling1D(pool_length = 2))
    model.add(LSTM(100, dropout_W = 0.2, dropout_W = 0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    print(model.summary())
    model.fit(x, Y, nb_epoch = 10, batch_size = 1000)
    models.append(("drop out CNN ", model))

    #LSTM with decay
    model = Sequential()
    model.add(LSTM(100))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    print (model.summary())
    model.fit(W, Y, nb_epoch = 10, batch_size = 1000)
    models.append(("Decay LSTM", model))
    
    #Non-decay LSTM
    x = sequence.pad_sequences(X)
    embedding_length = 300
    model = Sequential()
    model.add(Embedding(top_words, embedding_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    print (model.summary())
    model.fit(x, Y, nb_epoch = 10, batch_size = 1000)
    models.append(("Non-decay LSTM", model))
    
    #Non-decay LSTM with dropouts
    x = sequence.pad_sequences(X)
    embedding_length = 300
    model = Sequential()
    model.add(Embedding(top_words, embedding_length, dropout=0.2))
    #model.add(Dropout(0.2))
    model.add(LSTM(100, dropout_W = 0.2, dropout_U = 0.2))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    print (model.summary())    
    model.fit(x, Y, nb_epoch = 10, batch_size = 1000)
    models.append(("dropout LSTM", model))

    return (models)

def test (X, W, Y, model):
    pass

def modeling(conn, sentences, dz):
    #pts = pd.read_sql("SELECT DISTINCT SUBJECT_ID from UFM", conn)
    #pts =list(set(pts.SUBJECT_ID))
    #pool = []
    #for d in dz:
    #    pool += d.pos + d.neg

    decay = .9
    data = []; Data = []
    
    admits = pd.read_sql("SELECT * from admissions", conn)
   
    for d in dz:
        neg = random.sample(d.neg, len(d.pos))
        pts = d.pos+neg
        kf = KFold(n_splits = 5, shuffle = False)
        for train_index, test_index in kf.split(pts):
            #each train, test has format (s, time, hadm, 1/0)
            train, test = pts[train_index], pts[test_index]
            
            #make exclusion list for test patients
            introns = [t[0] for t in test]
            #instance = [t[2] for t in test]
            #sentences have format (s, hadm, [words], [times])
            lst = [i[2] for i in sentences if i[0] not in introns]
            
            #word2vec:
            #configure hyperparams as appropriate
            SG = gensim.models.Word2Vec(sentences = lst, sg = 1, size = 300, window = 10, min_count = 465, hs = 1, negative = 0, workers = 4)
            #CBOW = gensim.models.Word2Vec(sentences = lst, sg = 0, size = 300, window = 10, min_count = 465, hs = 1, negative = 0, workers = 4)

            #construct sequence feature from train(ing) set
            #X stands for raw feature input
            #W stands for word vectors from feature input trained by Word2Vec
            X_train = []; t_train = []; W_train = []; Y_train = []
            X_test = []; t_test = []; W_test = []; Y_test = []
            
            for t in train:
              
                #corpus is n x 2 tensor with each column containing the words and timing, respectively
                #only select the admission sequences which occur BEFORE the queried admission
                corpus = [[s[2], s[3]] for s in sentences if  (s[0] == t[0]) and (pd.to_datetime(admits[admits['HADM_ID']==s[1]].ADMITTIME.values[0]) <= t[1])]
                
                #order subject by time of entry for each sentence (admission)
                corpus = sorted(corpus, key = lambda x: x[1])
                
                #configure each timestamp to reflect time elapsed from first time entry
                #calculate time decay from initial event
                for item in range(len(corpus)):
                    corpus[item][1] = corpus[item][1] - corpus[0][1]
                #initial time entry is set to 0
                #e.g., corpus[0][1] = 0                
                
                #transpose into 2xn from nx2
                #this way, corpus[0] refers to words and corpus[1] refers to times
                corpus = list(map(list, zip(*corpus)))                  
                
                #corpus[0] are the word sequences, 1 x n (variable!) 
                #n is number of admissions (rows)
                #d is the number of events
                X_train.append(np.array(corpus[0]))
                #corpus[1] are the time sequences, also 1 x n (variable!)
                t_train.append(np.array(corpus[1]))
                #decay_factor is formulated as -lambda (decay) * time elapsed since intial time, an 1 x n vector
                decay_factor = np.array([math.exp(-1 * decay * elapse) for elapse in corpus[1]])
                
                #w is X projected into word vector form
                #results in n (variable!) x 300
                #w is elementwise operation on word vectors .* e ^ (decay_factor * time)
                w = np.multiply(np.array([SG[c] for c in corpus[0]]), decay_factor)
                #should now be 300x1
                W_train.append(w)
                #add label, which is last item of t in train
                Y_train.append(t[3])
                
            for t in test:
              
                #corpus is n x 2 tensor with each column containing the words and timing, respectively
                #only select the admission sequences which occur BEFORE the queried admission
                corpus = [[s[2], s[3]] for s in sentences if (s[0] == t[0]) and (pd.to_datetime(admits[admits['HADM_ID']==s[1]].ADMITTIME.values[0]) <= t[1])]
                
                #order subject by time of entry for each sentence (admission)
                corpus = sorted(corpus, key = lambda x: x[1])

                #configure each timestamp to reflect time elapsed from first time entry
                #calculate time decay from initial event
                for item in range(len(corpus)):
                    corpus[item][1] = corpus[item][1] - corpus[0][1]
                #initial time entry is set to 0
                corpus[0][1] = 0                
                
                #transpose into 2xn from nx2
                #this way, corpus[0] refers to words and corpus[1] refers to times
                corpus = list(map(list, zip(*corpus)))                
                
                #corpus[0] are the word sequences, 1 x n (variable!) 
                #n is number of admissions (rows)
                #d is the number of events
                X_test.append(np.array(corpus[0]))
                #corpus[1] are the time sequences, also 1 x n (variable!)
                t_test.append(np.array(corpus[1]))                
                #decay_factor is formulated as -lambda (decay) * time elapsed since intial time, an nx1 vector
                decay_factor = np.array([math.exp(-1 * decay * elapse) for elapse in corpus[1]])
                
                #w is elementwise operation on words .* e ^ (decay_factor * time)
                w = np.multiply(np.array([SG[c] for c in corpus[0]]), decay_factor)
                W_test.append(w)
                #add label, which is last item of t in train
                Y_test.append(t[3])
                
            #Y_train = list(map(lambda x: 1 if x[3] ==1 else 0, train))
            #Y_test = list(map(lambda x: 1 if x[3] ==1 else 0, test))

            #training
            models = training(X_train, W_train, Y_train)
            #testing
            for m in models:
                data.append(test(X_test, W_test, Y_test, m))
        Data.append(data)
            
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



##### DISEASES ######
#####################

#1) CHF = ['40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493', '4280', '428', '4281', '42820', '42822', '42830', '42832', '42840', '42842', '4289']
#2) Afib = ['42731', '4271', '42789']
#3) LMyc = ['1629']
#4) PTEN = ['1830', '193', '2330', '1930']
#5) Stroke = ['43491', '43411', '4349', '43401', '434', '4340', '43401', '43410','43490'] 
#6) Sepsis = ['99591', '99592', '0389', '0380','0381', '03811', '03812', '03819', '03810', '0382', '0383', '0384', '0388', '03840', '03841', '03842', '03843', '03844', 03849']
#7) RF = ['5845','5849','5856', '5846', '5847', '5848', '5851', '5852', '5853', '5854', '5855', '5859']
#8) Cirrhosis = ['5712','5715']
#9) T2DM = ['2500', '25000', '25001', '25002', '25003', '2501', '25010', '25011', '25012','25013', '2502', '25020', '25021', '25022', '25023', '25030', '2503', '25031', '25032', '25033', '25040', '25041', '25042', '25043', '2504', '2505', '25050', '25051', '25052', '25053', '2506', '25060', '25061', '25062', '25063', '2507', '25070', '25071', '25072', '25073', '2508', '25080', '25081', '25082', '25083', '2509', '25090', '25091', '25092', '25093']
#10) CAD = ['414', '4140','41400', '41401', '41402', '41403','41404','41405','41406','41407', '4141', '41410', '41411', '41412', '41419', '4142', '4143', '4144', '4148', '4149']
#11) ATH = ['4400', '44000', '4401', '44010', '4402', '44020', '4403', '44030', '4404', '44040','4408', '44080', '4409', '44090', '44021', '44022', '44023', '44024', '44029', '44031', '44032']
#12) ARDS = ['51881', '51884', '51883'] #EXCLUDING trauma/surgery related resp. failure.

##### Scratch Work #####
########################

#### EMBEDDING #####
   # corpus = []
   # count = 0
    
    #make patient corpus for disease querying purposes (slow!)
    #for p in pts:
    #    df = pd.read_sql("SELECT * from UFM2 where SUBJECT_ID = '%s'" %(p), conn)
    #    print ("+++++++++++")
    #    print ("Current Sess: {0}".format(count))
    #    print ("Preview:")
    #    print(df.head())
    #    pt = Patient(ufm_slice = df, library = lib)
    #    pt.Corpus()
    #    corpus.append([p, pt.corpus])
    #    count+=1
        
    #make corpus for disease specific purposes (fast!)
    #keys = [k[1] for k in lib]
    #count = 0; sentences=  []
    #for p in pts:
    #    df = pd.read_sql("SELECT * from UFM2 where SUBJECT_ID = '%s'" %(p), conn)
    #    print ("Current Sess: {0}".format(count))
    #    hadm = list(set(df.HADM_ID))
        #df = df.sort('TIME')
     #   for h in hadm:
     #       sentence = []
     #       for index, row in df[df['HADM_ID']==h].sort('TIME').iterrows():
     #           word = row['FEATURE']+'_' + str(row['DISCRETE_VALUE'])
     #           if word in keys: sentence.append(word)
     #       sentences.append((p, h, sentence))
     #   count +=1
        
