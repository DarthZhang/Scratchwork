# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 23:22:39 2016

@author: Andy
"""

import csv
import numpy as np

import pandas as pd
from pandas import DataFrame

import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from collections import Counter

df = pd.read_csv('C:/Users/Andy/Desktop/mimic/csv/DIAGNOSES_ICD.csv/DIAGNOSES_ICD_DATA_TABLE.csv')
key = pd.read_csv('C:/Users/Andy/Desktop/mimic/csv/DIAGNOSES_ICD.csv/D_ICD_DIAGNOSES_DATA_TABLE.csv')

CHF = ['40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493', '428.0', '4280', '428', '428.1','4281', '42820', '42822', '42830', '42832', '42840', '42842', '4289', '428.9']


#OK. Here's the plan:
#[1] Make a list of all the ICD9 CHF codes (there are many).
#[2] Use keys to search for them.
#[3] Make a filtered table from df using keys from keys
#key[key['ICD9_CODE'].isin(CHF)]

#[4] ID patients who have CHF codes.
patients = df[df['ICD9_CODE'].isin(CHF)]
#subject_IDs =patients["SUBJECT_ID"].unique() #unique patients with CHF
subjects = dict(Counter(patients["SUBJECT_ID"])) #creates Counter for each subjects 
cover2 = {i:j for (i,j) in subjects.items() if j>1} #filter subjects to find only patients with re-admissions for CHF

#To select for these CHF patients labevents, we first need  to buffer LABEVENTS
header = ["ROW_ID", "SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUE", "VALUENUM", "VALUEUOM", "FLAG"]
count =0
#dflabs = pd.DataFrame(columns=header)
#output = open("C:/Users/Andy/Desktop/mimic/csv/CHF ANALYSIS/CHF.csv",'a')
with open ("C:/Users/Andy/Desktop/mimic/csv/LABEVENTS.csv/LABEVENTS_DATA_TABLE.csv", buffering = 20000000) as f:
    for line in f:
        if count ==0:
            dflabs = pd.DataFrame(columns=header)
        else:
            data = [i.strip() for i in line.split(',')] #converts line (str) into list
            if data[3] in CHF: #checks if 'ICD_9' part of the input data is part of CHF screen
                print ("Picking up CHF screening just fine!")
                temp = dict(zip(header,data)) #makes data into a dictionary
                temp = pd.Series(temp) #converts data into Series
                dflabs=dflabs.append(temp, ignore_index=True) #adds Series data into dflabs, match by index = header
        count+=1
        
#output.close()

