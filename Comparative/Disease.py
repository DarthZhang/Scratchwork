# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:21:07 2017

@author: andy
"""

import sys, pickle
import os.path as path

import csv
import gzip
#import MySQLdb as mysql
import pymysql as mysql
import pandas as pd
from pandas import DataFrame
from pandas.io import sql as transfer


import numpy as np
import math
import datetime
#import matplotlib.pyplot as plt
import re

from scipy import stats
from collections import Counter
from itertools import combinations, tee
from datetime import date
from datetime import time
from datetime import timedelta
from sklearn import preprocessing

class Disease:
    #Input list of ICD9 code and MySQL connection. Output queries and +/- sample data depending on Onset/Readmission/Mortality. 
    
    def __init__ (self, dz, conn):
        self.dz = dz
        c = conn.cursor()
        
        #Diagnoses dataframe
        sql = "SELECT * from diagnoses"
        self.dx = pd.read_sql(sql=sql, con=conn)        
        
        #Subjects
        sql = "SELECT DISTINCT SUBJECT_ID FROM diagnoses WHERE ICD9_CODE in (%s)" % ','.join(['%s']*len(self.dz))
        self.subj = list(pd.read_sql(sql=sql, con=conn, params= tuple(self.dz)).SUBJECT_ID)
        
        #Admission Table for Subjects of Interest
        sql = "SELECT * from admissions WHERE SUBJECT_ID in (%s)" % ','.join(['%s']*len(self.subj))
        temp = [str(i) for i in self.subj]
        self.admits = pd.read_sql(sql=sql, con=conn, params = tuple(temp))

        c.close()
    
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
    
    def onset(self):
        self.queries = []
        for s in self.subj:
            hadm = list(self.admits[self.admits['SUBJECT_ID']==s].HADM_ID.values)
            t = [(pd.to_datetime(self.admits[self.admits['HADM_ID']==i]['ADMITTIME'].values[0]), i) for i in hadm]
            t = sorted(t)
            temp = sorted([i for i in t if not self.dx[(self.dx['HADM_ID']==i[1])&(self.dx['ICD9_CODE'].isin(self.dz))].empty])
            
            if len(temp) == len(t): 
                self.queries.append(s, temp[0][0], 0)
            else:
                self.queries.append(s, temp[0][0], 1)
    
        
    def readmission (self):
        self.pos = []
        self.neg = []
                        
        for s in self.subj:
            hadm = list(self.admits[self.admits['SUBJECT_ID']==s].HADM_ID.values)
            t = [(pd.to_datetime(self.admits[self.admits['HADM_ID']==i]['ADMITTIME'].values[0]), i) for i in hadm]
            t = sorted(t)
            
            pos = []
            neg=[]            
            
            if len(t) <2: 
                neg.append((s, t[0][0], t[0][1], 0))
                #print ((s,t[0][0], t[0][1]))
            else:   
                for t2,t1 in Disease.pairwise(iterable = reversed(t)):
                    if self.dx[(self.dx['HADM_ID']==t1[1])&(self.dx['ICD9_CODE'].isin(self.dz))].empty:
                        if self.dx[(self.dx['HADM_ID']==t2[1])&(self.dx['ICD9_CODE'].isin(self.dz))].empty:
                            pass
                        else: 
                            neg.append((s, t2[0], t2[1], 0))
                    elif (t2[0]-t1[0]).days >30:
                        neg.append((s, t1[0], t1[1], 0))
                    else:
                        pos.append((s, t1[0], t1[1], 1))
                        #print ((s, t1[0], t1[1]))
            
            if len(pos) >0:
                self.pos.append(pos[0])
            else:
                self.neg.append(neg[0])

                
    def mortality (self):
        self.queries = []
        dday = self.admits[self.admits.DEATHTIME.notnull()]
        dday=dday.loc[~dday['SUBJECT_ID'].duplicated()]
        schindler = list(set(self.subj) - set(dday.SUBJECT_ID))
        
        for index,row in dday.iterrows():
            if row['HAS_CHARTEVENTS_DATA']==1:
                self.queries.append((row['SUBJECT_ID'], pd.to_datetime(row['DEATHTIME']), 1))
        
        for s in schindler:
            hadm = list(self.admits[self.admits['SUBJECT_ID']==s].HADM_ID.values)
            H = hadm[-1]
            self.queries.append((s, pd.to_datetime(self.admits[self.admits['HADM_ID']==H]['ADMITTIME'].values[0]), 0))
        
        
        
#### SCRATCH WORK #####
                
            #temp = sorted([i for i in t if not self.dx[(self.dx['HADM_ID']==i[1])&(self.dx['ICD9_CODE'].isin(self.dz))].empty])
        
            #if len(temp) == 0: 
            #    self.zeroes +=1
            #elif len(temp) == 1: 
            #    self.ones +=1
            #    self.queries.append((s, temp[0][0], temp[0][0]+timedelta(days=30), 0))
            #else:
             #   self.many +=1
                #if(temp[-1] != t[-1]):
                #    self.queries.append((s, temp[-1][0], temp[-1][0]+timedelta(days=30), 0))
                #    self.pos.append(s)
                #elif (temp[-1][0] - temp[-2][0]).days > 30:
                
                #    self.exempt.append(s)
                #else:
                #    self.queries.append((s, temp[-2][0], temp[-1][0], 1))
                #    self.neg.append(s)




