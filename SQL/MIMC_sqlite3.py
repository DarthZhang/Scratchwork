# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 05:51:50 2016

@author: Andy
"""

import sqlite3
import csv
import pandas as pd
from pandas import DataFrame
from pandas.io import sql

import numpy as np
import math
import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from collections import Counter
from itertools import combinations
from datetime import date
from datetime import time
#import pymysql
#import pymysql.cursor
#conn= pymysql.connect(host='localhost',user='user',password='user',db='mimic.db',port='0034',cursorclass=pymysql.cursors.DictCursor)

conn = sqlite3.connect('mimic.db')
c = conn.cursor()

def create_tables():
    ## create tables ##
    c.execute('DROP TABLE IF EXISTS admissions')
    c.execute('DROP TABLE IF EXISTS diagnoses')
    c.execute('DROP TABLE IF EXISTS icds')
    c.execute('DROP TABLE IF EXISTS labevents')
    c.execute('DROP TABLE IF EXISTS procedureevents')
    c.execute('DROP TABLE IF EXISTS labitems')
    c.execute('CREATE TABLE IF NOT EXISTS admissions(ROW_ID INT, SUBJECT_ID INT, HADM_ID INT, ADMITTIME TIMESTAMP, DISCHTIME TIMESTAMP, DEATHTIME TIMESTAMP, ADMISSION_TYPE TEXT, ADMISSION_LOCATION TEXT, DISCHARGE_LOCATION TEXT, INSURANCE TEXT, LANGUAGE TEXT, RELIGION TEXT, MARITAL_STATUS TEXT, ETHNICITY TEXT, EDREGTIME TIMESTAMP, EDOUTTIME TIMESTAMP, DIAGNOSIS TEXT, HOSPITAL_EXPIRE_FLAG INT, HAS_IOEVENTS_DATA INT, HAS_CHARTEVENTS_DATA INT);')    
    #c.execute('CREATE TABLE IF NOT EXISTS chartevents')
    #c.execute('CREATE TABLE IF NOT EXISTS labitems')
    c.execute('CREATE TABLE IF NOT EXISTS diagnoses(ROW_ID INT, SUBJECT_ID INT, HADM_ID INT, SEQ_NUM INT, ICD9_CODE TEXT);')
    c.execute('CREATE TABLE IF NOT EXISTS icds(ROW_ID INT, ICD9_CODE TEXT, SHORT_TITLE TEXT, LONG_TITLE TEXT);')
    #c.execute('CREATE TABLE IF NOT EXISTS inputevents')
    #c.execute('CREATE TABLE IF NOT EXISTS outputevents')
    #c.execute('CREATE TABLE IF NOT EXISTS procedures')
    c.execute('CREATE TABLE IF NOT EXISTS labevents(ROW_ID INT, SUBJECT_ID INT, HADM_ID INT, ITEMID INT, CHARTTIME TIMESTAMP, VALUE TEXT, VALUENUM REAL, VALUEUOM TEXT, FLAG TEXT);')
    #c.execute('CREATE TABLE IF NOT EXISTS items')
    c.execute('CREATE TABLE IF NOT EXISTS procedureevents(ROW_ID INT, SUBJECT_ID INT, HADM_ID INT, ICUSTAY_ID INT, STARTTIME TIMESTAMP, ENDTIME TIMESTAMP, ITEMID INT, VALUE REAL, VALUEUOM TEXT, LOCATION TEXT, LOCATIONCATEGORY TEXT, STORETIME TIMESTAMP, CGID INT, ORDERID INT, LINKORDERID INT, ORDERCATEGORYNAME TEXT, SECONDARYORDERCATEGORYNAME TEXT, ORDERCATEGORYDESCRIPTION TEXT, ISOPENBAG INT, CONTINUEINNEXTDEPT INT, CANCELREASON INT, STATUSDESCRIPTION TEXT, COMMENTS_EDITEDBY TEXT, COMMENTS_CANCELEDBY TEXT, COMMENTS_DATE TIMESTAMP);')
    
    ##import admissions table
    with open('C:/Users/Andy/Desktop/mimic/csv/ADMISSIONS.csv/ADMISSIONS_DATA_TABLE.csv','r') as f:
        dr = csv.DictReader(f) #first line is read as header. ',' is delimiter.
        #to_db = [(i['c1'], i['c2'], i['c3'], i['c4'], i['c5'], i['c6'], i['c7'], i['c8'], i['c9'], i['c10'], i['c11'], i['c12'], i['c13'], i['c14'], i['c15'], i['c16'], i['c17'], i['c18'], i['c19'], i['c20']) for i in dr]
        to_db = [(i['ROW_ID'], i['SUBJECT_ID'], i['HADM_ID'], i['ADMITTIME'], i['DISCHTIME'], i['DEATHTIME'], i['ADMISSION_TYPE'], i['ADMISSION_LOCATION'], i['DISCHARGE_LOCATION'], i['INSURANCE'], i['LANGUAGE'], i['RELIGION'], i['MARITAL_STATUS'], i['ETHNICITY'], i['EDREGTIME'], i['EDOUTTIME'], i['DIAGNOSIS'], i['HOSPITAL_EXPIRE_FLAG'], i['HAS_IOEVENTS_DATA'], i['HAS_CHARTEVENTS_DATA']) for i in dr]
    c.executemany("INSERT INTO admissions(ROW_ID, SUBJECT_ID, HADM_ID, ADMITTIME, DISCHTIME, DEATHTIME, ADMISSION_TYPE, ADMISSION_LOCATION, DISCHARGE_LOCATION, INSURANCE, LANGUAGE, RELIGION, MARITAL_STATUS, ETHNICITY, EDREGTIME, EDOUTTIME, DIAGNOSIS, HOSPITAL_EXPIRE_FLAG, HAS_IOEVENTS_DATA, HAS_CHARTEVENTS_DATA) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);", to_db)
    conn.commit()
    
    #import diagnoses table
    with open('C:/Users/Andy/Desktop/mimic/csv/DIAGNOSES_ICD.csv/DIAGNOSES_ICD_DATA_TABLE.csv','r') as f:
        dr = csv.DictReader(f) #first line is read as header. ',' is delimiter.
        #to_db = [(i['c1'], i['c2'], i['c3'], i['c4'], i['c5'], i['c6'], i['c7'], i['c8'], i['c9'], i['c10'], i['c11'], i['c12'], i['c13'], i['c14'], i['c15'], i['c16'], i['c17'], i['c18'], i['c19'], i['c20']) for i in dr]
        to_db = [(i['ROW_ID'], i['SUBJECT_ID'], i['HADM_ID'], i['SEQ_NUM'], i['ICD9_CODE']) for i in dr]
    c.executemany("INSERT INTO diagnoses(ROW_ID, SUBJECT_ID, HADM_ID, SEQ_NUM, ICD9_CODE) VALUES (?, ?, ?, ?, ?);", to_db)
    conn.commit()
    
    #import icds
    with open('C:/Users/Andy/Desktop/mimic/csv/DIAGNOSES_ICD.csv/D_ICD_DIAGNOSES_DATA_TABLE.csv','r') as f:
        dr = csv.DictReader(f) #first line is read as header. ',' is delimiter.
        to_db = [(i['ROW_ID'], i['ICD9_CODE'], i['SHORT_TITLE'], i['LONG_TITLE']) for i in dr]
    c.executemany("INSERT INTO icds(ROW_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE) VALUES (?, ?, ?, ?);", to_db)
    conn.commit()
    
    #import labevents table
    in_csv = 'C:/Users/Andy/Desktop/mimic/csv/LABEVENTS.csv/LABEVENTS_DATA_TABLE.csv'
    chunksize = 100000
    with open(in_csv, 'r') as f:
        for numlines,l in enumerate (f): pass
    numlines +=1
    for i in range (0, numlines, chunksize) :
        dr = pd.read_csv(in_csv, header=None, nrows = chunksize, skiprows = i)
        columns = ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUE', 'VALUENUM', 'VALUEUOM', 'FLAG']
        dr.columns = columns
       # dtypes = {'ROW_ID': int, 'SUBJECT_ID': int, 'HADM_ID': int, 'ITEMID': int, 'CHARTTIME': time, 'VALUE': str, 'VALUENUM': float,  'FLAG': str}
        sql.to_sql(dr, name = 'labevents', con=conn, index=False, index_label = 'ROW_ID', if_exists = 'append')        
        print (i)
        #to_db = [(i['ROW_ID'], i['SUBJECT_ID'], i['HADM_ID'], i['ITEMID'], i['CHARTTIME'], i['VALUE'], i['VALUENUM'], i['VALUEUOM'], i['FLAG']) for i in dr]
    #c.executemany("INSERT INTO labevents(ROW_ID, SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUE, VALUENUM, VALUEUOM, FLAG) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);", to_db)
    conn.commit()
    
    #import procedure events table
    with open('C:/Users/Andy/Desktop/mimic/csv/PROCEDURES_ICD.csv/PROCEDUREEVENTS_MV_DATA_TABLE.csv','r') as f:
        dr = csv.DictReader(f) #first line is read as header. ',' is delimiter.
        to_db = [(i['ROW_ID'], i['SUBJECT_ID'], i['HADM_ID'], i['ICUSTAY_ID'], i['STARTTIME'], i['ENDTIME'], i['ITEMID'], i['VALUE'], i['VALUEUOM'], i['LOCATION'], i['LOCATIONCATEGORY'], i['STORETIME'], i['CGID'], i['ORDERID'], i['LINKORDERID'], i['ORDERCATEGORYNAME'], i['SECONDARYORDERCATEGORYNAME'], i['ORDERCATEGORYDESCRIPTION'], i['ISOPENBAG'], i['CONTINUEINNEXTDEPT'], i['CANCELREASON'], i['STATUSDESCRIPTION'], i['COMMENTS_EDITEDBY'], i['COMMENTS_CANCELEDBY'], i['COMMENTS_DATE']) for i in dr]
    c.executemany("INSERT INTO procedureevents(ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, STARTTIME, ENDTIME, ITEMID, VALUE, VALUEUOM, LOCATION, LOCATIONCATEGORY, STORETIME, CGID, ORDERID, LINKORDERID, ORDERCATEGORYNAME, SECONDARYORDERCATEGORYNAME, ORDERCATEGORYDESCRIPTION, ISOPENBAG, CONTINUEINNEXTDEPT, CANCELREASON, STATUSDESCRIPTION, COMMENTS_EDITEDBY, COMMENTS_CANCELEDBY, COMMENTS_DATE) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?, ?, ?, ?, ?,?, ?, ?, ?, ?,?, ?, ?, ?, ?);", to_db)
    conn.commit()



def get_CHF():
    df = pd.read_csv('C:/Users/Andy/Desktop/mimic/csv/DIAGNOSES_ICD.csv/DIAGNOSES_ICD_DATA_TABLE.csv')
    #key = pd.read_csv('C:/Users/Andy/Desktop/mimic/csv/DIAGNOSES_ICD.csv/D_ICD_DIAGNOSES_DATA_TABLE.csv')

    CHF = ['40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493', '428.0', '4280', '428', '428.1','4281', '42820', '42822', '42830', '42832', '42840', '42842', '4289', '428.9']
    
    patients = df[df['ICD9_CODE'].isin(CHF)]
    #subject_IDs =patients["SUBJECT_ID"].unique() #unique patients with CHF
    subjects = dict(Counter(patients["SUBJECT_ID"])) #creates Counter for each subjects 
    mult_adm = {i:j for (i,j) in subjects.items() if j>1}      #filter patients with multiple admissions for CHF
    no_re = {i:j for (i,j) in subjects.items() if j==1}     #filter patients with only one admission for CHF

    #split re-admit CHF patients into 180 day window 
    admissions =  pd.read_csv('C:/Users/Andy/Desktop/mimic/csv/ADMISSIONS.csv/ADMISSIONS_DATA_TABLE.csv')
    readm = admissions[admissions['SUBJECT_ID'].isin(mult_adm.keys())]      #multiple admits admissions data
    #re_dx = patients[patients['SUBJECT_ID'].isin(mult_adm.keys())]          #multiple admits diagnosis data

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
                if (not df[(df['HADM_ID']==i[0]) & (df['ICD9_CODE'].isin(CHF))].empty) & (not df[(df['HADM_ID']==i[1]) & (df['ICD9_CODE'].isin(CHF))].empty):
                    #This was a long statement. It basically checks if HADM_ID's of both readmissions within the 180 day window are CHF-related. 
                    y_positive.append([subj, i[0], i[1]])
                    
                    #[6] now split no re-admit CHF patients into those who those who deceased vs. those discharged (not deceased).
    no_re = admissions[admissions['SUBJECT_ID'].isin(no_re.keys())]
    deceased = no_re[no_re['HOSPITAL_EXPIRE_FLAG'] == 1]
    deceased = dict(Counter(deceased['SUBJECT_ID']))            #deceased patients on single admission for CHF
    discharged = no_re[no_re['HOSPITAL_EXPIRE_FLAG'] == 0]
    discharged = dict(Counter(discharged['SUBJECT_ID']))        #patients discharged with only one admission for CHF
    
    ypos = []
    for i in y_positive:
        if i[0] in ypos: pass
        else: ypos.append(i[0])
            
    yneg = []
    for i in mult_adm.keys():
        if i in ypos: pass
        else: yneg.append(i)
    yneg.append(no_re.keys())
    
    return (ypos, yneg)

def make_UFM():
    #CHF = ['40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493', '428.0', '4280', '428', '428.1','4281', '42820', '42822', '42830', '42832', '42840', '42842', '4289', '428.9']
    #sql = "SELECT * FROM diagnoses WHERE ICD9_CODE in ({seq}) GROUP BY SUBJECT_ID HAVING COUNT(*)>1".format(seq=','.join(['?']*len(CHF)))
    #sql2 = "SELECT * FROM diagnoses WHERE ICD9_CODE in ({seq}) GROUP BY SUBJECT_ID HAVING COUNT(*)==1".format(seq=','.join(['?']*len(CHF)))
    #c.execute(sql, CHF)
    #multi_adm = c.fetchall()  #this isolates CHF patients with multiple CHF diagnoses
    #c.execute(sql2, CHF) 
    #one_adm = c.fetchall()    #this isolates CHF patients with only one CHF diagnosis  
    #[print (row) for row in data]
    
    ypos, yneg = get_CHF()
    
    ypos = [str(i) for i in ypos]
    ypos1 = ypos[0:int(len(ypos)/2)]
    ypos2 = ypos[int(len(ypos)/2):]
    
    #Make UFM dataframe of Lab Value features
    #chunk_size = 10000
    #offset = 0
    #dfs = []
    #while True:
    #    sql = "SELECT SUBJECT_ID, HADM_ID, CHARTTIME AS 'TIME', ITEMID AS 'FEATURE', VALUE FROM labevents WHERE SUBJECT_ID in ypos LIMIT %d OFFSET %d " % (chunk_size, offset)        
   #     #sql = "SELECT SUBJECT_ID, HADM_ID, CHARTTIME AS 'TIME', ITEMID AS 'FEATURE', VALUE FROM labevents LIMIT %d OFFSET %d WHERE SUBECT_ID in ({seq})" % (chunk_size, offset).format(seq=','.join(['?']*len(ypos))) 
    #    dfs.append(sql.read_frame(sql, conn))
    #    offset += chunk_size
    #    if len(dfs[-1]) < chunk_size: break
    #df1 = pd.concat(dfs)
    
    sqla = "SELECT SUBJECT_ID, HADM_ID, CHARTTIME AS 'TIME', ITEMID AS 'FEATURE', VALUE FROM labevents WHERE SUBJECT_ID in ({seq})".format(seq=','.join(['?']*len(ypos1)))
    sqlb = "SELECT SUBJECT_ID, HADM_ID, CHARTTIME AS 'TIME', ITEMID AS 'FEATURE', VALUE FROM labevents WHERE SUBJECT_ID in ({seq})".format(seq=','.join(['?']*len(ypos2))) 
    dfa = pd.read_sql_query(sql=sqla, con = conn, params = ypos1)
    dfb = pd.read_sql_query(sql=sqlb, con = conn, params = ypos2)
    df1 = pd.concat([dfa, dfb])
    df1['FEATURE'] = 'l_' + df1['FEATURE'].astype(str)          #add 'l_' prefix to lab features
    print ("df1 made.")
    sql2a = "SELECT SUBJECT_ID, HADM_ID, STARTTIME AS 'TIME', ITEMID AS 'FEATURE', VALUE FROM procedureevents WHERE SUBJECT_ID in ({seq})".format(seq=','.join(['?']*len(ypos1)))
    sql2b = "SELECT SUBJECT_ID, HADM_ID, STARTTIME AS 'TIME', ITEMID AS 'FEATURE', VALUE FROM procedureevents WHERE SUBJECT_ID in ({seq})".format(seq=','.join(['?']*len(ypos2)))
    print ("df2 selected...")    
    dfa = pd.read_sql_query(sql=sql2a, con = conn, params = ypos1)
    dfb = pd.read_sql_query(sql=sql2b, con = conn, params = ypos2)
    df2 = pd.concat([dfa, dfb])
    df2['FEATURE'] = 'p_' + df2['FEATURE'].astype(str)          #add 'p_' prefix to procedure features
    print ("df2 made.")
    sql3a = "SELECT SUBJECT_ID, HADM_ID, ICD9_CODE AS 'FEATURE' FROM diagnoses WHERE SUBJECT_ID in ({seq})".format(seq=','.join(['?']*len(ypos1)))
    sql3b = "SELECT SUBJECT_ID, HADM_ID, ICD9_CODE AS 'FEATURE' FROM diagnoses WHERE SUBJECT_ID in ({seq})".format(seq=','.join(['?']*len(ypos2)))
    dfa = pd.read_sql_query(sql=sql3a, con = conn, params = ypos1)
    dfb = pd.read_sql_query(sql=sql3b, con = conn, params = ypos2)
    df3 = pd.concat([dfa, dfb])
    df3['FEATURE'] = 'd_' + df3['FEATURE'].astype(str)
    
    frames = [df1, df2]
    df = pd.concat(frames, keys=['l','p'])
    return (df1, df2, df3, df)
    
def data_entry():
    c.execute ('INSERT INTO t VALUES (x, y, z)')
    conn.commit()


#create_tables()
df1, df2, df3, df = make_UFM()
c.close()
conn.close()
