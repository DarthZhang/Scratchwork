# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 21:46:40 2016

@author: Andy
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 21:43:58 2016

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
count = 0
#output = open("C:/Users/Andy/Desktop/mimic/csv/LABEVENTS.csv/labs.csv",'a')
with open ("C:/Users/Andy/Desktop/mimic/csv/INPUTEVENTS.csv/INPUTEVENTS_CV_DATA_TABLE.csv", buffering = 20000000) as f:
    for line in f:
        if count<10:
            print (line)
            count+=1
        else:
            break
        
        #output.write(line)
#output.close()
#key = pd.read_csv('C:/Users/Andy/Desktop/mimic/csv/LABEVENTS.csv/D_ITEMS_DATA_TABLE.csv')
#key[key['ITEMID'].isin([50820,50800,50802,50804,50808])]
