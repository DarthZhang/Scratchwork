# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 23:22:39 2016

@author: Andy
"""

import csv
import numpy as np
import math

import pandas as pd
from pandas import DataFrame

import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from collections import Counter
from itertools import combinations
from datetime import date

df = pd.read_csv('C:/Users/Andy/Desktop/mimic/csv/DIAGNOSES_ICD.csv/DIAGNOSES_ICD_DATA_TABLE.csv')
key = pd.read_csv('C:/Users/Andy/Desktop/mimic/csv/DIAGNOSES_ICD.csv/D_ICD_DIAGNOSES_DATA_TABLE.csv')

CHF = ['40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493', '428.0', '4280', '428', '428.1','4281', '42820', '42822', '42830', '42832', '42840', '42842', '4289', '428.9']


#OK. Here's the plan:
#[1] Make a list of all the ICD9 CHF codes (there are many).
#[2] Use keys to search for them.
#[3] Make a filtered table from df using keys from keys
#key[key['ICD9_CODE'].isin(CHF)]

#[4] ID patients who have CHF codes, split into multiple CHF admission vs. singular CHF admission.
patients = df[df['ICD9_CODE'].isin(CHF)]
#subject_IDs =patients["SUBJECT_ID"].unique() #unique patients with CHF
subjects = dict(Counter(patients["SUBJECT_ID"])) #creates Counter for each subjects 
mult_adm = {i:j for (i,j) in subjects.items() if j>1}      #filter patients with multiple admissions for CHF
no_re = {i:j for (i,j) in subjects.items() if j==1}     #filter patients with only one admission for CHF

#[5] now split re-admit CHF patients into 180 day window 
admissions =  pd.read_csv('C:/Users/Andy/Desktop/mimic/csv/ADMISSIONS.csv/ADMISSIONS_DATA_TABLE.csv')
readm = admissions[admissions['SUBJECT_ID'].isin(mult_adm.keys())]      #multiple admits admissions data
re_dx = patients[patients['SUBJECT_ID'].isin(mult_adm.keys())]          #multiple admits diagnosis data

y_positive = []                             #This is the list that we will use to append Subject_IDs of CHF readmits within 180 day window.
                                            #Format is [Subject_ID, HADM_ID of first CHF incident, HADM_ID of readmission in 180 days].
for subj in mult_adm.keys():
    hadm = readm[readm['SUBJECT_ID']==subj]['HADM_ID']                  #for each patients, make list of HADM
    combos = list(combinations(hadm, 2))
    for i in combos:
        t1 = pd.to_datetime(readm[readm['HADM_ID']==i[0]]['ADMITTIME'])
        t2 = pd.to_datetime(readm[readm['HADM_ID']==i[1]]['ADMITTIME'])
        t1 = t1[t1.index[0]]                #convert into TIMESTAMP
        t2 = t2[t2.index[0]]                #convert into TIMESTAMP
        
        if abs((t1-t2).days) <=180:         #We evaluate the HADM_ID contents in windows of 180 days.
            if (not df[(df['HADM_ID']==i[1]) & (df['ICD9_CODE'].isin(CHF))].empty) & (not df[(df['HADM_ID']==i[1]) & (df['ICD9_CODE'].isin(CHF))].empty):
                #This was a long statement. It basically checks if HADM_ID's of both readmissions within the 180 day window are CHF-related. 
                y_positive.append([subj, i[0], i[1]])

#[6] now split no re-admit CHF patients into those who those who deceased vs. those discharged (not deceased).
no_re = admissions[admissions['SUBJECT_ID'].isin(no_re.keys())]
deceased = no_re[no_re['HOSPITAL_EXPIRE_FLAG'] == 1]
deceased = dict(Counter(deceased['SUBJECT_ID']))            #deceased patients on single admission for CHF
discharged = no_re[no_re['HOSPITAL_EXPIRE_FLAG'] == 0]
discharged = dict(Counter(discharged['SUBJECT_ID']))        #patients discharged with only one admission for CHF



############
#To select for these CHF patients in labevents, we first need  to buffer LABEVENTS
#header = ["ROW_ID", "SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUE", "VALUENUM", "VALUEUOM", "FLAG"]
#count =0
#dflabs = pd.DataFrame(columns=header)
#output = open("C:/Users/Andy/Desktop/mimic/csv/CHF ANALYSIS/CHF.csv",'a')
#with open ("C:/Users/Andy/Desktop/mimic/csv/LABEVENTS.csv/LABEVENTS_DATA_TABLE.csv", buffering = 20000000) as f:
#    for line in f:
#        if count ==0:
#            dflabs = pd.DataFrame(columns=header)
#        else:
#            data = [i.strip() for i in line.split(',')] #converts line (str) into list
#            if int(data[1]) in readm.keys(): #checks if 'SUBJECT_ID' part of the input data is part of CHF population
#                temp = dict(zip(header,data)) #makes data into a dictionary
#                temp = pd.Series(temp) #converts data into Series
#                dflabs=dflabs.append(temp, ignore_index=True) #adds Series data into dflabs, match by index = header
#        count+=1
        
#output.close()

