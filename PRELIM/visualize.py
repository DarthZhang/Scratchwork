# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:13:34 2016

@author: Andy
"""
import csv
import numpy as np

import pandas as pd
from pandas import DataFrame

import datetime

import matplotlib.pyplot as plt
#from matplotlib import style
#style.use('ggplot')

df = pd.read_csv('C:/Users/Andy/Desktop/mimic/csv/ADMISSIONS.csv/ADMISSIONS_DATA_TABLE.csv')
diagnoses = df['DIAGNOSIS']

#fill in missing data
for i in df['DIAGNOSIS']:
    if i!=i:
        df['DIAGNOSIS']=df['DIAGNOSIS'].fillna(value="MISSING")

#plotting SUBJECT_ID vs. DIAGNOSIS with Scatter Plot.
x,y = [i for i in df['SUBJECT_ID']],[j for j in df['DIAGNOSIS']]

plt.scatter(x,y, s=20, c='b')
plt.set(xticks=range(len(x)), xticklabels="SUBJECT ID",
        yticks=range(len(y)), yticklabels="DIAGNOSIS")
#plt.xticks (range(len(x)), x, align = 'center')
#plt.xyticks (range(len(y)), y, size = 'small')
plt.show()
