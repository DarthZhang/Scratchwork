# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 14:04:34 2017

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
from scipy import stats

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

##### Part 1. OOP labs #######
##############################
class Lab:
    
    count = 0
    
    def __init__(df):
        self.values = list(df['VALUE'])
        self.flags = list(df['FLAG'])
        self.hadm = list(df['HADM_ID'])
        self.subj = list(df['SUBJECT_ID'])
        self.t = list(df['TIME'])
        
    def is_number(s):
        try:
            float(s)
            return True
        except:
            return False
        
    def is_discrete (self):
        self.unique = list(set(self.values))
        digits = [x for x in self.unique if is_number(x)]
        if len(digits)/len(self.unique) > .95: 
            self.discrete= True
        else: 
            self.discrete = False
                                
    
    def clean_list(self):
        self.values = [float(x) for x in self.values if is_number(x)]
        if self.discrete == False:
            #regex
        
            c_to_d
        else:
            d_to_d
        
    def c_to_d (self):
        ranked = stats.rankdata(self.values)
        percentiles = ranked/len(values)*100
        bins = [0,20,40,60,80,100]
        self.values = np.digitize(percentiles, bins, right= True)
    
    def d_to_d (self):
        
        



##### Part 2. Discretize Function ######
########################################


def main():
    
    discretize()
    

def discretize():
    conn = mysql.connect(host = host, user = user, passwd = pw, db = mimic, port = port)
    c = conn.cursor()
    
    c.execute('DROP TABLE IF EXISTS Discrete_Labs')
    
    sql = "SELECT DISTINCT FEATURE from UFM where TYPE = '%s'" % ('l')
    lst = pd.read_sql(sql = sql, con = conn)
    lst = list(lst['FEATURE'])
    
    for i in lst:
        sql = "SELECT * from UFM where FEATURE = '%s'" % (i)
        df= pd.read_sql(sql = sql, con = conn)
        lab = Lab(df)
        
        # do stuff
   
 #Make the dataframe into SQL table:
 
        transfer.to_sql(df, name = 'Discete_Labs', con=conn, index=False, index_label = 'ROW_ID', if_exists = 'append', flavor = 'mysql')
           

    c.close()
    conn.close()
    
    print ("Labs are Discretized.")
    #return(store)


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
    
    
    
# SCRATCH WORK 
    
    #df = pd.read_csv(admissions_doc)
    #subjects = dict(Counter(df["SUBJECT_ID"])) #creates Counter for each unique subject
    #subj = list(subjects.keys())
    #subj = [str(i) for i in subj]

    #for s in subj:
    #    sql = "SELECT * from UFM where SUBJECT_ID = '%s' AND TYPE = '%s'" % (s, 'l')
    #    df = pd.read_sql_query(sql=sql, con = conn)
        