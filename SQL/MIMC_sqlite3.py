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
    c.execute('CREATE TABLE IF NOT EXISTS admissions(c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20);')    
    #c.execute('CREATE TABLE IF NOT EXISTS admissions (row, subject_id, ham_id, admittime, dischargetime, deathtime, adm_type, adm_loc, loc, insurance, lang, religion, marital, ethnic, EDreg, EDout, dx, expire, IO, charts);')
    c.execute('CREATE TABLE IF NOT EXISTS chartevents')
    c.execute('CREATE TABLE IF NOT EXISTS labevents')
    c.execute('CREATE TABLE IF NOT EXISTS diagnoses')
    c.execute('CREATE TABLE IF NOT EXISTS icds')
    c.execute('CREATE TABLE IF NOT EXISTS inputevents')
    c.execute('CREATE TABLE IF NOT EXISTS outputevents')
    c.execute('CREATE TABLE IF NOT EXISTS procedures')
    c.execute('CREATE TABLE IF NOT EXISTS labitems')
    c.execute('CREATE TABLE IF NOT EXISTS items')
    c.execute('CREATE TABLE IF NOT EXISTS procedureevents')
    
    ##import admissions table
    with open('"C:/Users/Andy/Desktop/mimic/csv/ADMISSIONS.csv/ADMISSIONS_DATA_TABLE.csv".csv','rb') as f:
        dr = csv.DictReader(f) #first line is read as header. ',' is delimiter.
        to_db = [(i['c1'], i['c2'], i['c3'], i['c4'], i['c5'], i['c6'], i['c7'], i['c8'], i['c9'], i['c10'], i['c11'], i['c12'], i['c13'], i['c14'], i['c15'], i['c16'], i['c17'], i['c18'], i['c19'], i['c20']) for i in dr]

    c.executemany("INSERT INTO admissions(c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);", to_db)
    conn.commit()
    
    

def data_entry():
    c.execute ('INSERT INTO t VALUES (x, y, z)')
    conn.commit()

def read_from_db():
    c.execute('SELECT * FROM table WHERE value=0 AND subject=0')
    data = c.fetchall()
    [print (row) for row in data]

#create_table()
#data_entry()

c.close()
conn.close()
