# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:26:53 2017

@author: af1tang
"""

import sys, pickle
import os.path as path

import csv
import gzip
import MySQLdb as mysql
import pandas as pd
from pandas import DataFrame
from pandas.io import sql as transfer


import numpy as np
import math
import datetime
import matplotlib.pyplot as plt

from collections import Counter
from itertools import combinations
from datetime import date
from datetime import time
from datetime import timedelta
from tempfile import mkdtemp


admissions_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/ADMISSIONS.csv.gz'
diagnoses_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/DIAGNOSES_ICD.csv.gz'
icds_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/D_ICD_DIAGNOSES.csv.gz'
procedures_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/PROCEDUREEVENTS_MV.csv.gz'
labevents_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/LABEVENTS.csv.gz'
items_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/D_ITEMS.csv.gz'
labitems_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/D_LABITEMS.csv.gz'
patients_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/PATIENTS.csv.gz'

###### Part 1. OOP Patient by UFM Data ########
###############################################

class Patient:
    
    count = 0
    
    def __init__(ufm_slice, dx_features, lab_features, proc_features, query):
        
        self.df = ufm_slice
        self.subj = list(set(self.df.SUBJECT_ID))[0]
        self.hadm = list(set(self.df.HADM_ID))
        self.t1 = query[1] #initial observation time
        self.t2 = query[2] #end observation time
        
        #epidemiological data
        self.birth = 90
        self.gender = 0
        self.race = 0
        self.marital = 0
        self.insurance = 0
        
        #initial features
        self.dxs = dx_features
        self.labs = lab_features
        self.procs = proc_features
        
        count +=1
    
        
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
    

    def make_feature_table (self):

    
        print ("Making Feature Table...")
        conn = mysql.connect (host=host, user=user, passwd=pw, db=mimic, port=port)
        c = conn.cursor()

        sparcity = []
        X = []
        Y = []
        sess = 0
    
        for query in self.queries:
            sess +=1
            s, t1, t2 = query[0], query[1], query[2]
            #timedelta determines how large to peak backwards for observation windows.
            lst = list(features.columns)
        
            #observation window: t1 - obs to t1. obs := {30, 60, 90, 180, 360, 720, inf}
            try:
                df = pd.read_hdf(ufm, str(s))
            except:
                sql = "SELECT * FROM ufm WHERE SUBJECT_ID == {0}".format(s)
                df = pd.read_sql_query(sql = sql, con = conn)
            df['TIME'] = pd.to_datetime(df['TIME'])
            mask = (df['TIME'] >= (t1-timedelta(days=obs))) & (df['TIME'] <=t1)
            df = (df.loc[mask])[(df.loc[mask])['SUBJECT_ID']==s]
        
            #take only ICD9 features in df
            df = df[df['FEATURE'].isin(lst)]
            df = df.sort('TIME', ascending = True)
        
            #print ("Currently on Session: {0} out of {1}.".format(sess, len(queries)))   
            #print ("DF size: {0}, features size: {1}".format(len(df), len(features)))
        
            temp = get_ICDs(df, features)
            x = temp.as_matrix()
            trials = len(x)
        
            #print ("Size of x: {0}".format(x.shape))
            temp = x.T
        
            try:
                temp = [sum(i) for i in temp]
            except:
                for j in range(0,len(temp)):
                    try: temp[j] = int(temp[j])
                    except: temp[j] = int(temp[j][0])
                temp = [sum(i) for i in temp]
            
            X.append(temp)
            #sparcity.append(1-(sum(temp)/(len(temp)*trials)))
            sparcity.append(1-(sum(temp)/(len(temp))))
        
            if query[3] == 1: Y.append(1)
            else: Y.append(0)
        
        c.close()
        conn.close()        
        
        print ("DONE!")
    
        return (X, Y, sparcity)

###### Part 2. Compilation #########
####################################

def main():
    print ("Do work in Main.")
    
def ICD9_features (subj):
    
    conn = sqlite3.connect(mimic_doc)
    c = conn.cursor()
    
    dx_f=[]
    
    count=0
    for s in subj:
        count+=1
        print ("Session {0} out of {1}".format(count,len(subj)))
        try:
            df = pd.read_hdf(ufm, str(s))
        except:
            sql = "SELECT * FROM ufm WHERE SUBJECT_ID == {0}".format(s)
            df = pd.read_sql_query(sql = sql, con = conn)
        dx = df[df['TYPE']=='d']['FEATURE']
        dx = list(set(filter(None,dx)))
        dx_f+=dx
        dx_f = list(set(dx_f))
    
    c.close()
    conn.close()            

    columns = list(set([str(i) for i in dx_f]))
    print (len(columns))
    
    features = pd.DataFrame(columns=columns)
    
    for key in columns:
        features.loc[0, key] = 0
        
    print (len(features))           
    
    return (features)


if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    desc = "Welcome to UFM Table Maker by af1tang."
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
    
    main()  