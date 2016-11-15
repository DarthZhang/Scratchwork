# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 17:23:57 2016

@author: af1tang
"""
import sys, pickle
import tensorflow as tf
import os.path as path

import sqlite3
import csv
import pandas as pd
from pandas import DataFrame
from pandas.io import sql


import numpy as np
import sklearn
import hmmlearn
import math
import datetime
import matplotlib.pyplot as plt

from collections import Counter
from itertools import combinations
from datetime import date
from datetime import time
from datetime import timedelta
from hmmlearn.hmm import GaussianHMM
from tempfile import mkdtemp

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LogisticRegression

admissions_doc = '/media/sf_mimic/csv/ADMISSIONS.csv/ADMISSIONS_DATA_TABLE.csv'
diagnoses_doc = '/media/sf_mimic/csv/DIAGNOSES_ICD.csv/DIAGNOSES_ICD_DATA_TABLE.csv'
icds_doc = '/media/sf_mimic/csv/DIAGNOSES_ICD.csv/D_ICD_DIAGNOSES_DATA_TABLE.csv'
procedures_doc = '/media/sf_mimic/csv/csv/PROCEDURES_ICD.csv/PROCEDUREEVENTS_MV_DATA_TABLE.csv'
labevents_doc = '/media/sf_mimic/csv/csv/LABEVENTS.csv/LABEVENTS_DATA_TABLE.csv'
items_doc = '/media/sf_mimic/csv/LABEVENTS.csv/D_ITEMS_DATA_TABLE.csv'
labitems_doc = '/media/sf_mimic/csv/LABEVENTS.csv/D_LABITEMS_DATA_TABLE.csv'
patients_doc = '/media/sf_mimic/csv/ADMISSIONS.csv/PATIENTS_DATA_TABLE.csv'
file_a = path.join(mkdtemp(), 'Xfiles.dat')
file_b = path.join(mkdtemp(), 'Yfiles.dat')
file_c = path.join(mkdtemp(), 'Zfiles.dat')
file_d = path.join (mkdtemp(), 'AAfiles.dat')

def main():
    
    #prediction window:
    Q180, Q360, Q450, Q720, subj = queries()
    try: 
        with open ("/media/sf_mimic/csv/CHF ANALYSIS/vars/ufm2.pickle","rb") as f:
            ufm=pickle.load(f)
        with open ("/media/sf_mimic/csv/CHF ANALYSIS/vars/Q180.pickle","rb") as f:
            Q180=pickle.load(f)
        with open ("/media/sf_mimic/csv/CHF ANALYSIS/vars/Q360.pickle","rb") as f:
            Q360=pickle.load(f)
        with open ("/media/sf_mimic/csv/CHF ANALYSIS/vars/Q450.pickle","rb") as f:
            Q450=pickle.load(f)
        with open ("/media/sf_mimic/csv/CHF ANALYSIS/vars/Q720.pickle","rb") as f:
            Q720=pickle.load(f)
    except: 
        Q180, Q360, Q450, Q720, y_positive, y_negative = queries()
        with open ("/media/sf_mimic/csv/CHF ANALYSIS/vars/flags.pickle","rb") as f:
            flags= pickle.load(f)
        ufm = make_ufm (subj, flags)
    
    #initialize features
    features = ICD9_features(ufm)
    
    #models, f1_scores, auc_scores, X, Y = LR_obs_windows(ufm, features, Q180)
    models, f1_scores, auc_scores, X, Y, lengths = LR_predict_windows (ufm, features, Q180, Q360, Q450, Q720)
        
    
    #return (Q180, Q360, Q450, Q720, y_positive, y_negative)
    return (models, f1_scores, auc_scores, X, Y, lengths)         
    
    

####################################################################
##### Part 1. INITIALIZEE UFMs. Skip if already exists. ############
####################################################################
  
def make_ufm (subj, flags):
   
    #connect to sql
    conn = sqlite3.connect('mimic.db')
    c = conn.cursor()
    
    subj = [str(i[0]) for i in subj]
    flags = [str(i) for i in flags]    
    
############# OPTION 1 ################
    cut = int(len(subj)/15)
    print ('\n'+"+++++++++ IN PROGRESS +++++++++")
    for i in range (0,15):
        s = subj[i*cut:((i+1)*cut)]
        print ("Cycle number: {0}, Offset: {1}, Chunk: {2}".format(i, i*cut, (i+1)*cut))
        if i == 14:
            s = subj[i*cut:]
            print ("Ignore above, actual chunksize is {0} to end.".format(i*cut))

        sql_lab = "SELECT SUBJECT_ID, HADM_ID, CHARTTIME AS 'TIME', ITEMID AS 'FEATURE', VALUE, FLAG FROM labevents WHERE SUBJECT_ID IN ({0}) AND ITEMID IN ({1})".format(','.join('?'*len(s)),','.join('?'*len(flags)))
        sql_proc = "SELECT SUBJECT_ID, HADM_ID, STARTTIME AS 'TIME', ITEMID AS 'FEATURE', VALUE FROM procedureevents WHERE SUBJECT_ID IN ({0})".format(','.join('?'*len(s)))
        sql_dx = "SELECT SUBJECT_ID, HADM_ID, ICD9_CODE AS 'FEATURE' FROM diagnoses WHERE SUBJECT_ID IN ({0})".format(','.join('?'*len(s)))
        sql_dx2 = "SELECT HADM_ID, ADMITTIME AS 'TIME' FROM admissions WHERE SUBJECT_ID IN ({0})".format(','.join('?'*len(s)))
        
        lab_params = tuple(s+flags)
        other_params = tuple(s)
       
        df_lab = pd.read_sql_query(sql=sql_lab, con = conn, params = lab_params)
        df_proc = pd.read_sql_query(sql=sql_proc, con = conn, params = other_params)    
        df_dx1 = pd.read_sql_query(sql=sql_dx, con = conn, params = other_params)
        df_dx2 = pd.read_sql_query(sql=sql_dx2, con=conn, params = other_params)
        df_dx= pd.merge(df_dx1, df_dx2, how = 'outer', on = 'HADM_ID')
        
        df_dx['VALUE'] = 1
        df_dx['FLAG'] = None
        df_proc['FLAG'] = None
        df_lab['TYPE'] = 'l'
        df_proc['TYPE'] = 'p'
        df_dx['TYPE'] = 'd'
        
        if (i ==0): 
            frames = [df_lab, df_proc, df_dx]
            df = pd.concat(frames)
        else:
            frames = [df, df_lab, df_proc, df_dx]
            df = pd.concat(frames)

    #Make the dataframe into SQL table:
    from pandas.io import sql
    c.execute('DROP TABLE IF EXISTS CHF_dataframe')
    sql.to_sql(df, name = 'CHF_dataframe', con=conn, index=False, index_label = 'ROW_ID', if_exists = 'append')
    
    c.close()
    conn.close()
    
    print ("UFM TABLES COMPLETE!")
    
    return (df)
 
 ##############################################################
 ##### Part 2. Making Queries Based On Observation Windows#####
 ##############################################################

def queries (dx = diagnoses_doc, adm = admissions_doc):
    #make CHF filter
    df = pd.read_csv(dx)
    admissions = pd.read_csv(adm)
    CHF = ['40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493', '428.0', '4280', '428', '428.1','4281', '42820', '42822', '42830', '42832', '42840', '42842', '4289', '428.9']
    
    print ("Loading data...")    
    
    patients = df[df['ICD9_CODE'].isin(CHF)]
    subjects = dict(Counter(patients["SUBJECT_ID"])) #creates Counter for each unique subject
    subj = list(subjects.keys())
    admits = admissions[admissions['SUBJECT_ID'].isin(subj)]  #finds these patients in admissions table  

    #prediction_window sizes: 0, 180, 360, 450, 720 from index date of prediction
    y_pos180 = []
    y_pos360 = []
    y_pos450 = []
    y_pos720 = []
    
    y_neg180 = []
    y_neg360 = []
    y_neg450 = []
    y_neg720 = []    
    
    print ("Total number of subjects with CHF: {0}".format(len(subj)))
    
    sess=0
    for s in subj:

        sess+=1
        print ("Session number: {0}".format(sess))        
        
        hadm = admits[admits['SUBJECT_ID']==s]['HADM_ID']
        
        #check length number of admissions per s: if <2 but is CHF related, add to y_neg
        H = list(pd.Series(hadm).values)
        if len(H)==0: pass
        elif (len(H)==1) & (not df[(df['HADM_ID']==H[0]) & (df['ICD9_CODE'].isin(CHF))].empty): 
            y_neg180.append([s, H[0], -1])
            y_neg360.append([s, H[0], -1])
            y_neg450.append([s, H[0], -1])
            y_neg720.append([s, H[0], -1])
   
        t = [(pd.to_datetime(admits[admits['HADM_ID']==i]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i]['ADMITTIME']).index[0]],i) for i in H]
        t = sorted(t)
        
        combos = list(combinations(t,2))
        
        for i in combos:
            difference = i[1][0] - i[0][0]
            if (difference.days) <=180: 
                if (not df[(df['HADM_ID']==i[0][1]) & (df['ICD9_CODE'].isin(CHF))].empty) & (not df[(df['HADM_ID']==i[1][1]) & (df['ICD9_CODE'].isin(CHF))].empty): 
                    y_pos180.append([s,i[0][1], i[1][1]])
                else: 
                    y_neg180.append([s,i[0][1], i[1][1]])
                    
            elif (180<difference.days<=360):
                if (not df[(df['HADM_ID']==i[0][1]) & (df['ICD9_CODE'].isin(CHF))].empty) & (not df[(df['HADM_ID']==i[1][1]) & (df['ICD9_CODE'].isin(CHF))].empty): 
                    y_pos360.append([s,i[0][1], i[1][1]])
                else: 
                    y_neg360.append([s,i[0][1], i[1][1]])
            
            elif (360<difference.days<=450):
                if (not df[(df['HADM_ID']==i[0][1]) & (df['ICD9_CODE'].isin(CHF))].empty) & (not df[(df['HADM_ID']==i[1][1]) & (df['ICD9_CODE'].isin(CHF))].empty): 
                    y_pos450.append([s,i[0][1], i[1][1]])
                else: 
                    y_neg450.append([s,i[0][1], i[1][1]])
                    
            elif (450<difference.days<=720):
                if (not df[(df['HADM_ID']==i[0][1]) & (df['ICD9_CODE'].isin(CHF))].empty) & (not df[(df['HADM_ID']==i[1][1]) & (df['ICD9_CODE'].isin(CHF))].empty): 
                    y_pos720.append([s,i[0][1], i[1][1]])
                else: 
                    y_neg720.append([s,i[0][1], i[1][1]])
                    
            elif (difference.days>720): 
                pass
            

    #Step 2. Make Prediction Windows for X
    print ("====================================")
    print ("Making Prediction Windows for X...")
    print ("...")
    print ("...")
    
    t_neg180=[]
    t_neg360=[]
    t_neg450=[]
    t_neg720=[]
        
    t_pos180 = [(i[0], pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME']).index[0]], pd.to_datetime(admits[admits['HADM_ID']==i[2]]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i[2]]['ADMITTIME']).index[0]]) for i in y_pos180]    
    t_pos360 = [(i[0], pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME']).index[0]], pd.to_datetime(admits[admits['HADM_ID']==i[2]]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i[2]]['ADMITTIME']).index[0]]) for i in y_pos360]    
    t_pos450 = [(i[0], pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME']).index[0]], pd.to_datetime(admits[admits['HADM_ID']==i[2]]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i[2]]['ADMITTIME']).index[0]]) for i in y_pos450]    
    t_pos720 = [(i[0], pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME']).index[0]], pd.to_datetime(admits[admits['HADM_ID']==i[2]]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i[2]]['ADMITTIME']).index[0]]) for i in y_pos720]    

    for i in y_neg180:
        time1 = pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME']).index[0]]
        if i[2] == -1:
            time2 = time1+timedelta(days=180)
        else:
            time2 = pd.to_datetime(admits[admits['HADM_ID']==i[2]]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i[2]]['ADMITTIME']).index[0]]
        t_neg180.append((i[0], time1, time2))

    for i in y_neg360:
        time1 = pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME']).index[0]]
        if i[2] == -1:
            time2 = time1+timedelta(days=180)
        else:
            time2 = pd.to_datetime(admits[admits['HADM_ID']==i[2]]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i[2]]['ADMITTIME']).index[0]]
        t_neg360.append((i[0], time1, time2))

    for i in y_neg450:
        time1 = pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME']).index[0]]
        if i[2] == -1:
            time2 = time1+timedelta(days=180)
        else:
            time2 = pd.to_datetime(admits[admits['HADM_ID']==i[2]]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i[2]]['ADMITTIME']).index[0]]
        t_neg450.append((i[0], time1, time2))
        
    for i in y_neg720:
        time1 = pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME']).index[0]]
        if i[2] == -1:
            time2 = time1+timedelta(days=180)
        else:
            time2 = pd.to_datetime(admits[admits['HADM_ID']==i[2]]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i[2]]['ADMITTIME']).index[0]]
        t_neg720.append((i[0], time1, time2))
    
    t_neg180 = [(i[0],i[1],i[2], 0) for i in t_neg180]
    t_neg360 = [(i[0],i[1],i[2], 0) for i in t_neg360]
    t_neg450 = [(i[0],i[1],i[2], 0) for i in t_neg450]
    t_neg720 = [(i[0],i[1],i[2], 0) for i in t_neg720]    

    t_pos180 = [(i[0],i[1],i[2], 1) for i in t_pos180]
    t_pos360 = [(i[0],i[1],i[2], 1) for i in t_pos360]
    t_pos450 = [(i[0],i[1],i[2], 1) for i in t_pos450]
    t_pos720 = [(i[0],i[1],i[2], 1) for i in t_pos720]

    Q180 = t_pos180+t_neg180
    Q180 = sorted(Q180, key=lambda element: (element[0], element[1]))
    Q360 = t_pos360+t_neg360
    Q360 = sorted(Q360, key=lambda element: (element[0], element[1]))
    Q450 = t_pos450+t_neg450
    Q450 = sorted(Q450, key=lambda element: (element[0], element[1]))
    Q720 = t_pos720+t_neg720
    Q720 = sorted(Q720, key=lambda element: (element[0], element[1]))

    y_pos = y_pos180+ y_pos360+y_pos450 + y_pos720
    y_neg = y_neg180 + y_neg360 + y_neg450 + y_neg720
  
    return (Q180, Q360, Q450, Q720, subj)


def ICD9_features (ufm):
    
    dx_f = ufm[ufm['TYPE']=='d']['FEATURE']
    dx_f=list(filter(None,dx_f))
    columns = list(set([str(i) for i in dx_f]))
    print (len(columns))
    
    features = pd.DataFrame(columns=columns)
    
    for key in columns:
        features.loc[0, key] = 0
        
    print (len(features))           #now a (1,4122) dataframe
    
    return (features)
    
def get_ICDs(df, f):
    features = f.copy(deep = True)
    first_time = 0
    time = 0
    
    for index, row in df.iterrows():
        if first_time == 0:
            t_prev = row['TIME']
            features.loc[time, str(row['FEATURE'])] = row['VALUE']
            first_time+=1
        else:
            t_curr = row['TIME']
            if (t_prev ==t_curr):
                features.loc[time, str(row['FEATURE'])] = row['VALUE']
            else:
                t_prev = row['TIME']
                time+=1
                features.loc[time] = features.loc[time-1]
                features.loc[time, str(row['FEATURE'])] = row['VALUE']
    
    return (features)
    

def make_feature_table (ufm, queries, features, obs):
    
    #make timestamps readable
    ufm['TIME'] = pd.to_datetime(ufm['TIME'])
    
    X = []
    Y = []
    sess = 0
    
    for query in queries:
        sess +=1
        s, t1, t2 = query[0], query[1], query[2]
        #timedelta determines how large to peak backwards for observation windows.
        lst = list(features.columns)        
        
        #observation window: t1 - obs to t1. obs := {30, 60, 90, 180, 360, 720, inf}
        mask = (ufm['TIME'] >= (t1-timedelta(days=obs))) & (ufm['TIME'] <=t1)
        df = (ufm.loc[mask])[(ufm.loc[mask])['SUBJECT_ID']==s]
        
        #take only ICD9 features in df
        df = df[df['FEATURE'].isin(lst)]
        df = df.sort('TIME', ascending = True)
        
        #print ("Currently on Session: {0} out of {1}.".format(sess, len(queries)))   
        #print ("DF size: {0}, features size: {1}".format(len(df), len(features)))
        
        temp = get_ICDs(df, features)
        x = temp.as_matrix()
        #print ("Size of x: {0}".format(x.shape))
        temp = x.T
        temp = [sum(i) for i in temp]
        X.append(temp)
        
        if query[3] == 1: Y.append(1)
        else: Y.append(0)
        
    return (X, Y)
    
######################################
## Part 3. Logistic Regression ###### 
#####################################

def LR_obs_windows(ufm, features, queries):
    
    #obs_window sizes: 30, 90, 180, 360, 450, 720, and inf days
    obs = [30, 90, 180, 360,450,720, 99999]
    #x, y = make_feature_table(ufm, Q180, features, 180)
    
    models = []
    auc_scores = []
    f1_scores = []
    #Y = []
    prev = 0
    
    for i in obs:
        
        print("=============")
        
        x, y= make_feature_table (ufm, queries, features, i)
        x = np.array(x)
        y = np.array(y)
        
        #appending x to X, y to Y for future references
        b = np.memmap(file_b, mode = 'w+', shape = (x.shape[0], x.shape[1]), dtype = 'object')
        b[:] = x
        c = np.memmap(file_c, mode = 'w+', shape = (y.shape[0], y.shape[1]), dtype = 'object')
        c[:] = y
        if prev == 0:
            a = np.memmap(file_a, mode = 'w+', shape = (x.shape[0], x.shape[1]), dtype='object')
            a[:] = b[:]
            d = np.memmap(file_d, mode = 'w+', shape = (y.shape[0], y.shape[1]), dtype='object')
            d[:] = c[:]
            prev+=len(x)
        else:
            X = np.memmap(file_a, mode = 'r+', shape = (prev+len(x), x.shape[1]), dtype='object')
            X[prev:, : ] = b[:]
            Y = np.memmap(file_d, mode = 'r+', shape = (prev+len(y), y.shape[1]), dtype='object')
            Y[prev:, : ] = c[:]
            
            prev+=len(x)
        
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size = .2)
        model = LogisticRegression(n_jobs = -1)
        model.fit(x_train, y_train)
        models.append(model)
        
        print ("{0} obs_window model qualities:".format(i))
        print ("Accuracy: {0}".format(model.score(x_test, y_test)))
        
        y_predicted = model.predict(x_test)
        auc_score = roc_auc_score(y_test, y_predicted, average = 'macro')
        f1 = f1_score (y_test, y_predicted, average = 'macro')
        auc_scores.append(auc_score)
        f1_scores.append(f1)
        
        print ("AUC score is {0}".format(auc_score))
        print ("F1 score is {0}".format(f1))
    
    return (models, f1_scores, auc_scores, X, Y)    



def LR_predict_windows (ufm, features, Q180, Q360, Q450, Q720):
    
    lst = [Q180, Q360, Q450, Q720]    
    models = []
    auc_scores = []
    f1_scores = []
    prev = 0
    sess = 0
    Y = []
    lengths = []
    
    for i in lst:
        x, y= make_feature_table (ufm, i, features, 720)
        x = np.array(x)
        lengths.append(x.shape[0])
        
        #appending x to X, y to Y for future references
        b = np.memmap(file_b, mode = 'w+', shape = (x.shape[0], x.shape[1]), dtype = 'object')
        b[:] = x
        Y.append(y)
    
        if prev == 0:
            a = np.memmap(file_a, mode = 'w+', shape = (x.shape[0], x.shape[1]), dtype='object')
            a[:] = b[:]
            prev+=len(x)
        else:
            X = np.memmap(file_a, mode = 'r+', shape = (prev+len(x), x.shape[1]), dtype='object')
            X[prev:, : ] = b[:]
            prev+=len(x)
        
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size = .2)
        model = LogisticRegression(n_jobs = -1)
        model.fit(x_train, y_train)
        models.append(model)
        sess+=1
        
        print ("{0} Prediction Window qualities:".format(sess*180))
        print ("Accuracy: {0}".format(model.score(x_test, y_test)))
    
        y_predicted = model.predict(x_test)
        auc_score = roc_auc_score(y_test, y_predicted, average = 'macro')
        f1 = f1_score (y_test, y_predicted, average = 'macro')
        auc_scores.append(auc_score)
        f1_scores.append(f1)
        
        print ("AUC score is {0}".format(auc_score))
        print ("F1 score is {0}".format(f1))    
    
    return (models, f1_scores, auc_scores, X, Y, lengths)    
    
###### PLOTTING ###########
###########################
    
def obs_window_plotting (df):
    df.set_value(6,'Obs Window Size', 999)
    df['Obs Window Size'] = df['Obs Window Size'].apply(pd.to_numeric)
    df = df.sort('Obs Window Size')
    x = list(df['Obs Window Size'])
    acc = list(df['Accuracies'])
    f1 = list(df['F1 Score'])
    auc = list(df['ROC_AUC'])
    
    plt.subplot(221)
    plt.scatter(x,acc)
    plt.plot(x,acc)
    #plt.xlabel('obs window')
    plt.ylabel('accuracy')
    plt.title('Accuracy')
    plt.show()

    plt.subplot(222)
    plt.scatter(x,auc)
    plt.plot(x,auc)
    plt.show()
    plt.xlabel('obs window')
    plt.ylabel('auc roc')
    plt.title('ROC AUC scores')

    plt.subplot(223)
    plt.scatter(x,f1)
    plt.plot(x,f1)
    plt.xlabel('obs window')
    plt.ylabel('f1 scores')
    plt.title('F1 Scores')
    plt.show()
    #%matplotlib qt

def prediction_window_plotting (df):
    df['Prediction Window'] = df['Prediction Window'].apply(pd.to_numeric)
    df = df.sort('Prediction Window')
    x = list(df['Prediction Window'])
    acc = list(df['Accuracies'])
    f1 = list(df['F1 Score'])
    auc = list(df['ROC_AUC'])
    
    plt.subplot(221)
    plt.scatter(x,acc)
    plt.plot(x,acc)
    #plt.xlabel('prediction window')
    plt.ylabel('accuracy')
    plt.title('Accuracy')
    plt.show()

    plt.subplot(222)
    plt.scatter(x,auc)
    plt.plot(x,auc)
    plt.show()
    plt.xlabel('prediction window')
    plt.ylabel('auc roc')
    plt.title('ROC AUC scores')

    plt.subplot(223)
    plt.scatter(x,f1)
    plt.plot(x,f1)
    plt.xlabel('prediction window')
    plt.ylabel('f1 scores')
    plt.title('F1 Scores')
    plt.show()
    #%matplotlib qt
    
###################
 
if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    desc = "Welcome to CHF Readmission Predictor by af1tang."
    version = "version 0.1"
    opt = OptionParser (description = desc, version=version)
    opt.add_option ('-i', action = 'store', type ='string', dest='input', help='Please input path to Database File.')
    opt.add_option ('-o', action = 'store', type = 'string', dest='output', default='CHF_data.pickle', help='Please state desired storage file for this session.')
    (cli, args) = opt.parse_args()
    opt.print_help()
    
    models, f1_scores, auc_scores, X, Y, lengths= main()
    #Q180, Q360, Q450, Q720, y_pos, y_neg = main()