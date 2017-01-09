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


###### OOP Patient with UFM Data ########
#########################################

class Patient:
    
    count = 0
    
    def __init__(ufm_slice, query):
        
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
        #self.dxs = dx_features
        #self.labs = lab_features
        #self.procs = proc_features
        #self.dx_sparcity = 0
        #self.label = 0
        
        count +=1
        
##### Part 1. Initialize dx_features, lab_features, and proc_features #####
###########################################################################

    def dx_features (self):
 
        dx = self.df[self.df['TYPE']=='d']['FEATURE']
        dx = list(set(filter(None,dx)))           

        columns = list(set([str(i) for i in dx]))    
        self.dxs = pd.DataFrame(columns=columns)
    
        for key in columns:
            self.dxs.loc[0, key] = 0
                 

##### Part 2. Make Feature Table #####
######################################
        
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
    
        #timedelta determines how large to peak backwards for observation windows.
        lst = list(self.dxs.columns)
        
        #observation window: t1 - obs to t1. obs := 720 days
        
        self.df['TIME'] = pd.to_datetime(self.df['TIME'])
        mask = (self.df['TIME'] >= (self.t1-timedelta(days=720))) & (self.df['TIME'] <=self.t1)
        self.df = (self.df.loc[mask])[(self.df.loc[mask])['SUBJECT_ID']==self.subj]
        
        #take only ICD9 features in df
        self.df = self.df[self.df['FEATURE'].isin(lst)]
        self.df = self.df.sort('TIME', ascending = True)
        
        #print ("Currently on Session: {0} out of {1}.".format(sess, len(queries)))   
        #print ("DF size: {0}, features size: {1}".format(len(df), len(features)))
        
        temp = self.get_ICDs(self.df, self.dxs)
        x = temp.as_matrix()
        
        #print ("Size of x: {0}".format(x.shape))
        temp = x.T
        
        try:
            temp = [sum(i) for i in temp]
        except:
            for j in range(0,len(temp)):
                try: temp[j] = int(temp[j])
                except: temp[j] = int(temp[j][0])
            temp = [sum(i) for i in temp]
            
        self.dx_sparcity = (1-(sum(temp)/(len(temp))))
        
        if self.query[3] == 1: self.label = 1
        else: self.label = 0
        
