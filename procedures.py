# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 19:24:34 2016

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


df = pd.read_csv('C:/Users/Andy/Desktop/mimic/csv/PROCEDURES_ICD.csv/PROCEDUREEVENTS_MV_DATA_TABLE.csv')
key = pd.read_csv('C:/Users/Andy/Desktop/mimic/csv/PROCEDURES_ICD.csv/PROCEDURES_ICD_DATA_TABLE.csv')