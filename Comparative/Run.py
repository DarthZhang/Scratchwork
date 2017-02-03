# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:38:42 2017

@author: andy
"""

import sys, pickle
import os.path as path

import csv
import gzip
#import MySQLdb as mysql
import pymysql as mysql
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from pandas import DataFrame
from pandas.io import sql as transfer


import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
import re

from scipy import stats
from collections import Counter
from itertools import combinations
from datetime import date
from datetime import time
from datetime import timedelta
from sklearn import preprocessing


##### FILE LOCATIONS ######
###########################

admissions_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/ADMISSIONS.csv.gz'
diagnoses_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/DIAGNOSES_ICD.csv.gz'
icds_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/D_ICD_DIAGNOSES.csv.gz'
procedures_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/PROCEDUREEVENTS_MV.csv.gz'
labevents_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/LABEVENTS.csv.gz'
items_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/D_ITEMS.csv.gz'
labitems_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/D_LABITEMS.csv.gz'
patients_doc = '/mnt/research/data/MIMIC3/physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_3/PATIENTS.csv.gz'

##### Import Dependencies ######
################################

import DatabaseBuilder as DB
import Disease
from Discretize import Lab
import Patient


def main():
    
    flags = make_database(conn)
    lib = discretize(conn, flags)
    querying(conn)
    indexes = embedding(conn, lib)
    modeling()

def make_database(conn):
    DB.make_sql(conn, admissions_doc, diagnoses_doc, icds_doc, procedures_doc, labevents_doc, items_doc, labitems_doc, patients_doc)
    DB.make_ufm(conn, engine)
    DB.make_demo(conn, engine)
    flags = DB.filter_labs(conn)
    return (flags)

def discretize(conn, flags):
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS Discrete_Labs')
    
    sql = "SELECT DISTINCT FEATURE from UFM where TYPE = '%s'" % ('l')
    #sql = "SELECT DISTINCT FEATURE from UFM"
    lst = pd.read_sql(sql = sql, con = conn)
    lst = list(lst['FEATURE'])
    
    count = 0
    l_values = []
    
    for i in lst:
        sql = "SELECT * from UFM where FEATURE = '%s'" % (i)
        df= pd.read_sql(sql = sql, con = conn)
        lab = Lab(df, flags)
        lab.is_discrete()
        lab.clean_list()
        l_values.append([i,lab.max])
        count+=1
        #Make the dataframe into SQL table:
        transfer.to_sql(lab.df, name = 'D_UFM', con=engine, index=False, index_label = 'ROW_ID', if_exists = 'append')
        print ("++++++++++++++++++++++")
        print ("Session: {0}".format(count))
        print (lab.df.head())
    
    df2= pd.read_sql("SELECT * from UFM where TYPE = 'p' or TYPE = 'd'", conn)
    df2['DISCRETE_VALUE'] = 1
    transfer.to_sql(df2, name = 'D_UFM', con=engine, index=False, index_label = 'ROW_ID', if_exists = 'append')

    n_values = [[x,1] for x in list(set(df2.FEATURE))]
    
    print ("UFM Table is Discretized.")
    
    values = l_values + n_values
    lib =[]; count = 0
    for v in values:
        if np.isnan(v[1]) or v[1]<1:
            pass
        elif v[1] == 1: 
            lib.append((count, v[0]+'_'+'1', 0))
            count+=1
        else:
            for i in range(0, v[1]+1): 
                lib.append((count, v[0] + '_' + str(i), 0))
                count+=1
    c.close()
    return (lib)

    
def querying(conn):
    chf = ['40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493', '4280', '428', '4281', '42820', '42822', '42830', '42832', '42840', '42842', '4289']
    afib = ['42731', '4271', '42789']
    lmyc = ['1629']
    pten = ['1830', '193', '2330', '1930']
    stroke = ['43491', '43411', '4349', '43401', '434', '4340', '43401', '43410','43490'] 
    sepsis = ['99591', '99592', '0389', '0380','0381', '03811', '03812', '03819', '03810', '0382', '0383', '0384', '0388', '03840', '03841', '03842', '03843', '03844', '03849']
    rf = ['5845','5849','5856', '5846', '5847', '5848', '5851', '5852', '5853', '5854', '5855', '5859']
    cirrhosis = ['5712','5715']
    t2dm = ['2500', '25000', '25001', '25002', '25003', '2501', '25010', '25011', '25012','25013', '2502', '25020', '25021', '25022', '25023', '25030', '2503', '25031', '25032', '25033', '25040', '25041', '25042', '25043', '2504', '2505', '25050', '25051', '25052', '25053', '2506', '25060', '25061', '25062', '25063', '2507', '25070', '25071', '25072', '25073', '2508', '25080', '25081', '25082', '25083', '2509', '25090', '25091', '25092', '25093']
    cad = ['414', '4140','41400', '41401', '41402', '41403','41404','41405','41406','41407', '4141', '41410', '41411', '41412', '41419', '4142', '4143', '4144', '4148', '4149']
    ath = ['4400', '44000', '4401', '44010', '4402', '44020', '4403', '44030', '4404', '44040','4408', '44080', '4409', '44090', '44021', '44022', '44023', '44024', '44029', '44031', '44032']
    ards = ['51881', '51884', '51883'] #EXCLUDING trauma/surgery related resp. failure.

    CHF = Disease(chf, conn)
    Afib = Disease(afib, conn)
    LMYC = Disease(lmyc, conn)
    PTEN = Disease(pten, conn)
    STROKE = Disease(stroke, conn)
    SEPSIS = Disease(sepsis, conn)
    RF = Disease(rf, conn)
    CIRRHOSIS = Disease(cirrhosis, conn)
    T2DM = Disease(t2dm, conn)
    CAD = Disease(cad, conn)
    ATH = Disease(ath, conn)
    ARDS = Disease(ards, conn)
    
    CHF.readmission()
    Afib.readmission()
    LMYC.readmission()
    PTEN.readmission()
    STROKE.readmission()
    SEPSIS.readmission()
    RF.readmission()
    CIRRHOSIS.readmission()
    T2DM.readmission()
    CAD.readmission()
    ATH.readmission()
    ARDS.readmission()
    

def embedding(conn, lib):
    pts = pd.read_sql("SELECT DISTINCT SUBJECT_ID from UFM", conn)
    pts =list(set(pts.SUBJECT_ID))
    indexes = []
    count = 0
    for p in pts:
        df = pd.read_sql("SELECT * from UFM2 where SUBJECT_ID = '%s'" %(p), conn)
        print ("+++++++++++")
        print ("Current Sess: {0}".format(count))
        print ("Preview:")
        print(df.head())
        corpus = Patient(ufm_slice = df, library = lib)
        corpus.Corpus()
        indexes.append([p, corpus.corpus])
        count+=1
    
    return (indexes)

    
def modeling():
    model = Model()


if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    desc = "Welcome to PipeLiner by af1tang."
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
    
    #connect to MySQL using engine to write pandas df --> mysql
    conn = mysql.connect(host = host, user = user, passwd = pw, db = mimic, port = port)    
    engine = create_engine ("mysql+pymysql://af1tang:illidan@illidan-gpu-1.egr.msu.edu:3306/MIMIC3")
    
    main()  
    
    conn.close()


##### DISEASES ######
#####################

#1) CHF = ['40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493', '4280', '428', '4281', '42820', '42822', '42830', '42832', '42840', '42842', '4289']
#2) Afib = ['42731', '4271', '42789']
#3) LMyc = ['1629']
#4) PTEN = ['1830', '193', '2330', '1930']
#5) Stroke = ['43491', '43411', '4349', '43401', '434', '4340', '43401', '43410','43490'] 
#6) Sepsis = ['99591', '99592', '0389', '0380','0381', '03811', '03812', '03819', '03810', '0382', '0383', '0384', '0388', '03840', '03841', '03842', '03843', '03844', 03849']
#7) RF = ['5845','5849','5856', '5846', '5847', '5848', '5851', '5852', '5853', '5854', '5855', '5859']
#8) Cirrhosis = ['5712','5715']
#9) T2DM = ['2500', '25000', '25001', '25002', '25003', '2501', '25010', '25011', '25012','25013', '2502', '25020', '25021', '25022', '25023', '25030', '2503', '25031', '25032', '25033', '25040', '25041', '25042', '25043', '2504', '2505', '25050', '25051', '25052', '25053', '2506', '25060', '25061', '25062', '25063', '2507', '25070', '25071', '25072', '25073', '2508', '25080', '25081', '25082', '25083', '2509', '25090', '25091', '25092', '25093']
#10) CAD = ['414', '4140','41400', '41401', '41402', '41403','41404','41405','41406','41407', '4141', '41410', '41411', '41412', '41419', '4142', '4143', '4144', '4148', '4149']
#11) ATH = ['4400', '44000', '4401', '44010', '4402', '44020', '4403', '44030', '4404', '44040','4408', '44080', '4409', '44090', '44021', '44022', '44023', '44024', '44029', '44031', '44032']
#12) ARDS = ['51881', '51884', '51883'] #EXCLUDING trauma/surgery related resp. failure.


