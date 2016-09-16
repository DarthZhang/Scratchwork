# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 16:56:27 2016

@author: Andy
"""

import csv
import numpy as np

import pandas as pd
from pandas import DataFrame

import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#from matplotlib import style
#style.use('ggplot')

from collections import Counter
#the Ultimate Frequency Counter of lists.


df = pd.read_csv('C:/Users/Andy/Desktop/mimic/csv/DIAGNOSES_ICD.csv/DIAGNOSES_ICD_DATA_TABLE.csv')
key = pd.read_csv('C:/Users/Andy/Desktop/mimic/csv/DIAGNOSES_ICD.csv/D_ICD_DIAGNOSES_DATA_TABLE.csv')

df.head()
key[key['ICD9_CODE'].isin(['E8790','42789','42822','4263','41401'])]
#filter key by specific ICD9_CODE

x,y = [i for i in df['SUBJECT_ID']],[j for j in df['ICD9_CODE']]
xcount = Counter(x)
ycount = Counter(y)

#BAR GRAPH:
#plt.bar(range(len(ycount)), ycount.values(), align='center')
#plt.xticks(range(len(ycount)), list(ycount.keys()))
#plt.xlabel("ICD9 Diagnosis")
#plt.ylabel("Volume of Patients")

#plt.show()

#PIE CHART:
groups = []
count = []
others = 0
cols = ['#191970','#001CF0','#0038E2','#0055D4','#0071C6','#008DB8','#00AAAA','#00C69C','#00E28E','#00FF80',]

for i in ycount:
    if ycount[i]<3900:
        others += ycount[i]
    else:
        groups.append(i)
        count.append(ycount[i])
        
#groups.append('others')
#count.append(others)

#plt.pie(count, labels = groups, colors = cols, autopct='%1.1f%%')
#plt.title("Top 20 ICD9 Diagnoses by Frequency")

#Histoplot with BINS:
labels, values = zip(*ycount.items())
indexes = np.arange(len(labels))
plt.bar(indexes, values, .3)
plt.xlabel("ICD9 Diagnosis")
plt.ylabel("Volume of Patients")

plt.show()