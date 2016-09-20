# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 05:51:50 2016

@author: Andy
"""

import sqlite3
import csv
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib as plt

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
    with open('C:/Users/Andy/Desktop/mimic/csv/LABEVENTS.csv/LABEVENTS_DATA_TABLE.csv','r') as f:
        dr = csv.DictReader(f) #first line is read as header. ',' is delimiter.
        to_db = [(i['ROW_ID'], i['SUBJECT_ID'], i['HADM_ID'], i['ITEMID'], i['CHARTTIME'], i['VALUE'], i['VALUENUM'], i['VALUEUOM'], i['FLAG']) for i in dr]
    c.executemany("INSERT INTO labevents(ROW_ID, SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUE, VALUENUM, VALUEUOM, FLAG) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);", to_db)
    conn.commit()
    
    #import procedure events table
    with open('C:/Users/Andy/Desktop/mimic/csv/PROCEDURES_ICD.csv/PROCEDUREEVENTS_MV_DATA_TABLE.csv','r') as f:
        dr = csv.DictReader(f) #first line is read as header. ',' is delimiter.
        to_db = [(i['ROW_ID'], i['SUBJECT_ID'], i['HADM_ID'], i['ICUSTAY_ID'], i['STARTTIME'], i['ENDTIME'], i['ITEMID'], i['VALUE'], i['VALUEUOM'], i['LOCATION'], i['LOCATIONCATEGORY'], i['STORETIME'], i['CGID'], i['ORDERID'], i['LINKORDERID'], i['ORDERCATEGORYNAME'], i['SECONDARYORDERCATEGORYNAME'], i['ORDERCATEGORYDESCRIPTION'], i['ISOPENBAG'], i['CONTINUEINNEXTDEPT'], i['CANCELREASON'], i['STATUSDESCRIPTION'], i['COMMENTS_EDITEDBY'], i['COMMENTS_CANCELEDBY'], i['COMMENTS_DATE']) for i in dr]
    c.executemany("INSERT INTO procedureevents(ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, STARTTIME, ENDTIME, ITEMID, VALUE, VALUEUOM, LOCATION, LOCATIONCATEGORY, STORETIME, CGID, ORDERID, LINKORDERID, ORDERCATEGORYNAME, SECONDARYORDERCATEGORYNAME, ORDERCATEGORYDESCRIPTION, ISOPENBAG, CONTINUEINNEXTDEPT, CANCELREASON, STATUSDESCRIPTION, COMMENTS_EDITEDBY, COMMENTS_CANCELEDBY, COMMENTS_DATE) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?, ?, ?, ?, ?,?, ?, ?, ?, ?,?, ?, ?, ?, ?);", to_db)
    conn.commit()


def data_entry():
    c.execute ('INSERT INTO t VALUES (x, y, z)')
    conn.commit()

def read_from_db():
    CHF = ['40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493', '428.0', '4280', '428', '428.1','4281', '42820', '42822', '42830', '42832', '42840', '42842', '4289', '428.9']
    sql = "SELECT * FROM diagnoses WHERE ICD9_CODE in ({seq}) GROUP BY SUBJECT_ID HAVING COUNT(*)>1".format(seq=','.join(['?']*len(CHF)))
    sql2 = "SELECT * FROM diagnoses WHERE ICD9_CODE in ({seq}) GROUP BY SUBJECT_ID HAVING COUNT(*)==1".format(seq=','.join(['?']*len(CHF)))
    c.execute(sql, CHF)
    multi_adm = c.fetchall()
    c.execute(sql2, CHF)
    one_adm = c.fetchall()
    #[print (row) for row in data]


create_tables()
#read_from_db()
c.close()
conn.close()
