# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 16:42:16 2016

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
#from matplotlib import style
#style.use('ggplot')
count = 0
items = []
#df = pd.DataFrame({'ROW_ID':[], 'SUBJECT_ID':[], "HADM_ID": [],"ITEMID":[],"CHARTTIME":[],"VALUE":[],"VALUENUM":[],"VALUEUOM":[],"FLAG":[]})
#output = open("C:/Users/Andy/Desktop/mimic/csv/LABEVENTS.csv/labs.csv",'a')
with open ("C:/Users/Andy/Desktop/mimic/csv/LABEVENTS.csv/LABEVENTS_DATA_TABLE.csv", buffering = 20000000) as f:
    for line in f:
        #if count<10:
        #    print (line)
        #    count+=1
        #else:
        #    break
        
        #itemid freq counter:        
        if count ==0:
            count+=1
        else:
            items.append(line.split(',')[3])
            count+=1

c= Counter()
c=Counter(items)
lst = c.most_common()
for i in range(0,10):
    print (lst[i][0])
        #output.write(line)
#output.close()
        
#get dictionary keys:        
key = pd.read_csv('C:/Users/Andy/Desktop/mimic/csv/LABEVENTS.csv/D_ITEMS_DATA_TABLE.csv')
labkeys = pd.read_csv('C:/Users/Andy/Desktop/mimic/csv/LABEVENTS.csv/D_LABITEMS_DATA_TABLE.csv')
labkeys[labkeys['ITEMID'].isin([51221,50971,50983,50912,50902, 51006, 50882, 51265, 50868, 51301])]

#construct whole dataframe:
#df = 
#for line in pd.read_csv('C:/Users/Andy/Desktop/mimic/csv/LABEVENTS.csv/LABEVENTS_DATA_TABLE.csv', chunksize= 200000000):
    