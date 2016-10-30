import sys, pickle
import tensorflow as tf
import os.path as path

import sqlite3
import csv
import pandas as pd
from pandas import DataFrame
from pandas.io import sql


import numpy as np
import sklearn
import hmmlearn
import math
import datetime

from collections import Counter
from itertools import combinations
from datetime import date
from datetime import time
from datetime import timedelta
from hmmlearn.hmm import GaussianHMM
from tempfile import mkdtemp

admissions_doc = '/media/sf_mimic/csv/ADMISSIONS.csv/ADMISSIONS_DATA_TABLE.csv'
diagnoses_doc = '/media/sf_mimic/csv/DIAGNOSES_ICD.csv/DIAGNOSES_ICD_DATA_TABLE.csv'
icds_doc = '/media/sf_mimic/csv/DIAGNOSES_ICD.csv/D_ICD_DIAGNOSES_DATA_TABLE.csv'
procedures_doc = '/media/sf_mimic/csv/csv/PROCEDURES_ICD.csv/PROCEDUREEVENTS_MV_DATA_TABLE.csv'
labevents_doc = '/media/sf_mimic/csv/csv/LABEVENTS.csv/LABEVENTS_DATA_TABLE.csv'
items_doc = '/media/sf_mimic/csv/LABEVENTS.csv/D_ITEMS_DATA_TABLE.csv'
labitems_doc = '/media/sf_mimic/csv/LABEVENTS.csv/D_LABITEMS_DATA_TABLE.csv'
patients_doc = '/media/sf_mimic/csv/ADMISSIONS.csv/PATIENTS_DATA_TABLE.csv'
file_a = path.join(mkdtemp(), 'Xfiles.dat')
file_b = path.join(mkdtemp(), 'Yfiles.dat')

def main():
	#sess = tf.InteractiveSession()
	#tf.initialize_all_variables().run()
	#sess.run()
     
     
	print ('We will be using three different strategies for CHF readmission prediction.')

def feature_table(dx=diagnoses_doc, adm = admissions_doc):
    
    #make CHF filter
    df = pd.read_csv(dx)
    admissions = pd.read_csv(adm)
    CHF = ['40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493', '428.0', '4280', '428', '428.1','4281', '42820', '42822', '42830', '42832', '42840', '42842', '4289', '428.9']
    
    patients = df[df['ICD9_CODE'].isin(CHF)]
    subjects = dict(Counter(patients["SUBJECT_ID"])) #creates Counter for each unique subject
    subj = list(subjects.keys())
    admits = admissions[admissions['SUBJECT_ID'].isin(subj)]  #finds these patients in admissions table  

    #Step 1. Label Y
    y_positive = []
    y_negative = []
    
    for s in subj:
        hadm = admits[admits['SUBJECT_ID']==s]['HADM_ID']
        
        #check length number of admissions per s: if <2 but is CHF related, add to y_neg
        H = list(pd.Series(hadm).values)
        if len(H)==0: pass
        elif (len(H)==1) & (not df[(df['HADM_ID']==H[0]) & (df['ICD9_CODE'].isin(CHF))].empty): y_negative.append([s, H[0], -1])
        
        #t = [pd.to_datetime(admits[admits['HADM_ID']==i]['ADMITTIME']) for i in H]
        #t = [i[i.index[0]] for i in t]
        t = [(pd.to_datetime(admits[admits['HADM_ID']==i]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i]['ADMITTIME']).index[0]],i) for i in H]
        t = sorted(t)
        
        combos = list(combinations(t,2))
        
        for i in combos:
            difference = i[1][0] - i[0][0]
            if (difference.days) <=180: 
                if (not df[(df['HADM_ID']==i[0][1]) & (df['ICD9_CODE'].isin(CHF))].empty) & (not df[(df['HADM_ID']==i[1][1]) & (df['ICD9_CODE'].isin(CHF))].empty): 
                    y_positive.append([s,i[0][1], i[1][1]])
                else:
                    y_negative.append([s,i[0][1], i[1][1]])

        

        #combos = list(combinations(hadm, 2)) 
        #for i in combos:
        #    t1 = pd.to_datetime(admits[admits['HADM_ID']==i[0]]['ADMITTIME'])
        #    t2 = pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME'])
        #    t1 = t1[t1.index[0]]                #convert into TIMESTAMP
        #    t2 = t2[t2.index[0]]                #convert into TIMESTAMP
            
        #    if abs((t1-t2).days) <=180:         #We evaluate the HADM_ID contents in windows of 180 days.
        #        if (not df[(df['HADM_ID']==i[0]) & (df['ICD9_CODE'].isin(CHF))].empty) & (not df[(df['HADM_ID']==i[1]) & (df['ICD9_CODE'].isin(CHF))].empty):
        #            #This was a long statement. It basically checks if HADM_ID's of both readmissions within the 180 day window are CHF-related. 
        #            if (t1-t2).days <0:                    
        #                y_positive.append([s,i[1],i[0]])
        #            else: y_positive.append([s, i[0], i[1]])
        #        else: 
        #            if (t1-t2).days<0:
        #                y_negative.append([s,i[1],i[0]])
        #            else: y_negative.append([s, i[0], i[1]])
    
    #Querying patients based on Y
    #queries = []
    #for i in y_positive:
    #    queries. append((i,1))
    #for j in y_negative:
    #    queries.append((j,0))
    
    print ("=====================================")
    print ("\n"+"Number of unique Subjects with CHF diagnoses: {0}".format(len(subjects)))
    print ("Number of CHF readmission cases (within 180 day window): {0}".format(len(y_positive)))
    print ("Total number of CHF admissions to work with: {0}".format(len(queries)))
    
    #Step 2. Make Feature Table for X
    print ("====================================")
    print ("Making feature table for X...")
    print ("...")
    print ("...")

    t_pos = []
    t_neg = []
    
    t_pos = [(i[0], pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME']).index[0]], pd.to_datetime(admits[admits['HADM_ID']==i[2]]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i[2]]['ADMITTIME']).index[0]]) for i in y_positive]    
    
    for i in y_negative:
        time1 = pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i[1]]['ADMITTIME']).index[0]]
        if i[2] == -1:
            time2 = time1+timedelta(days=180)
        else:
            time2 = pd.to_datetime(admits[admits['HADM_ID']==i[2]]['ADMITTIME'])[pd.to_datetime(admits[admits['HADM_ID']==i[2]]['ADMITTIME']).index[0]]
        t_neg.append((i[0], time1, time2))
    
    #PICKLE: t_pos, t_neg, t_queries
    #feature_table = x_features(y_positive, y_negative, t_pos, t_neg)
    t_neg = [(i[0],i[1],i[2], 0) for i in t_neg]
    t_pos = [(i[0],i[1],i[2], 1) for i in t_pos]
    t_queries = t_pos+t_neg
    t_queries = t_queries = sorted(t_queries, key=lambda element: (element[0], element[1]))
    
    print ("Complete!") 
        
    
    #Save this for future use.
    #pickle_out = open("/media/sf_mimic/csv/CHF ANALYSIS/vars/CHF_data.pickle", "wb")
    #pickle.dump(y_positive, y_negative, queries, pickle_out)
    #pickle_out.close()
        
    #with open ("CHF_data.pickle","rb") as f:
    #   ypos, yneg, queries = pickle.load(f)  
    
    
  
    return (df, admissions, patients, subjects, admits, y_positive, y_negative)
    

def x_features(y_positive, y_negative, flags, t_pos, t_neg):
    #make query table
    ypos = [(item[0], item[1]) for item in y_positive]
    yneg = [(item[0], item[1]) for item in y_negative]
    ypos = set(ypos)
    yneg = set(yneg)
    difference = yneg - ypos 
    queries = list(ypos) + list(difference)

    #connect to sql
    conn = sqlite3.connect('mimic.db')
    c = conn.cursor()
    
    #1. Grab all the relevant lab features from labevents -- those that are flaggable.
    #sql_flags = "SELECT ITEMID, count(ITEMID) FROM labevents WHERE FLAG <> ' ' GROUP BY ITEMID"
    #flag_labs = pd.read_sql_query(sql = sql_flags, con= conn)
    #flags = list(flag_labs['ITEMID'])
    #PICKLE THIS!
    #pickle_out = open("/media/sf_mimic/csv/CHF ANALYSIS/vars/flags.pickle","wb")
    #pickle.dump(flags, pickle_out)
    #pickle_out.close()
    
    #OPTIONAL (for option2): initialize sql commands.
    #sql_lab = "SELECT SUBJECT_ID, HADM_ID, CHARTTIME AS 'TIME', ITEMID AS 'FEATURE', VALUE FROM labevents WHERE SUBJECT_ID = {0} AND HADM_ID = {1} AND ITEMID IN ({2})".format('?', '?',','.join('?'*len(flags)))
    #sql_proc = "SELECT SUBJECT_ID, HADM_ID, STARTTIME AS 'TIME', ITEMID AS 'FEATURE', VALUE FROM procedureevents WHERE SUBJECT_ID = ? AND HADM_ID = ?"
    #sql_dx = "SELECT SUBJECT_ID, HADM_ID, ICD9_CODE AS 'FEATURE' FROM diagnoses WHERE SUBJECT_ID = ? AND HADM_ID = ?"
    #sql_dx2 = "SELECT HADM_ID, ADMITTIME AS 'TIME' FROM admissions WHERE SUBJECT_ID = ? AND HADM_ID = ?"

    subj = [i[0] for i in queries]
    hadm = [i[1] for i in queries]
    subj = [str(i) for i in subj]
    hadm = [str(i) for i in hadm]
    flags = [str(i) for i in flags]    
    
############# OPTION 1 ################
    cut = int(len(subj)/15)
    print ('\n'+"+++++++++ IN PROGRESS +++++++++")
    for i in range (0,15):
        s = subj[i*cut:((i+1)*cut)]
        h = hadm[i*cut:((i+1)*cut)]
        print ("Cycle number: {0}, Offset: {1}, Chunk: {2}".format(i, i*cut, (i+1)*cut))
        if i == 14:
            s = subj[i*cut:]
            h = hadm[i*cut:]
            print ("Ignore above, actual chunksize is {0} to end.".format(i*cut))

        sql_lab = "SELECT SUBJECT_ID, HADM_ID, CHARTTIME AS 'TIME', ITEMID AS 'FEATURE', VALUE, FLAG FROM labevents WHERE SUBJECT_ID IN ({0}) AND HADM_ID IN ({1}) AND ITEMID IN ({2})".format(','.join('?'*len(s)), ','.join('?'*len(h)),','.join('?'*len(flags)))
        sql_proc = "SELECT SUBJECT_ID, HADM_ID, STARTTIME AS 'TIME', ITEMID AS 'FEATURE', VALUE FROM procedureevents WHERE SUBJECT_ID IN ({0}) AND HADM_ID IN ({1})".format(','.join('?'*len(s)), ','.join('?'*len(h)))
        sql_dx = "SELECT SUBJECT_ID, HADM_ID, ICD9_CODE AS 'FEATURE' FROM diagnoses WHERE SUBJECT_ID IN ({0}) AND HADM_ID IN ({1})".format(','.join('?'*len(s)), ','.join('?'*len(h)))
        sql_dx2 = "SELECT HADM_ID, ADMITTIME AS 'TIME' FROM admissions WHERE SUBJECT_ID IN ({0}) AND HADM_ID IN ({1})".format(','.join('?'*len(s)), ','.join('?'*len(h)))
        
        lab_params = tuple(s+h+flags)
        other_params = tuple(s+h)
       
        df_lab = pd.read_sql_query(sql=sql_lab, con = conn, params = lab_params)
        df_proc = pd.read_sql_query(sql=sql_proc, con = conn, params = other_params)    
        df_dx1 = pd.read_sql_query(sql=sql_dx, con = conn, params = other_params)
        df_dx2 = pd.read_sql_query(sql=sql_dx2, con=conn, params = other_params)
        df_dx= pd.merge(df_dx1, df_dx2, how = 'outer', on = 'HADM_ID')
        
        df_dx['VALUE'] = 1
        df_dx['FLAG'] = None
        df_proc['FLAG'] = None
        df_lab['TYPE'] = 'l'
        df_proc['TYPE'] = 'p'
        df_dx['TYPE'] = 'd'
        
        if (i ==0): 
            frames = [df_lab, df_proc, df_dx]
            df = pd.concat(frames)
        else:
            frames = [df, df_lab, df_proc, df_dx]
            df = pd.concat(frames)



##########  OPTION 2 #######################
 #   count = 0  
 #   for query in queries:
 #       a = str(query[0])
 #       b = str(query[1])
        
 #       lab_params = tuple([a]+[b]+flags)
 #       df_lab = pd.read_sql_query(sql=sql_lab, con = conn, params = lab_params)
 #       df_proc = pd.read_sql_query(sql=sql_proc, con = conn, params = (a, b))    
 #       df_dx1 = pd.read_sql_query(sql=sql_dx, con = conn, params = (a, b))
 #       df_dx2 = pd.read_sql_query(sql=sql_dx2, con=conn, params = (a,b))
 #       df_dx= pd.merge(df_dx1, df_dx2, how = 'outer', on = 'HADM_ID')
        
 #       df_dx['VALUE'] = 1
 #       df_lab['TYPE'] = 'l'
 #       df_proc['TYPE'] = 'p'
 #       df_dx['TYPE'] = 'd'
        
 #       if (count ==0): 
 #           frames = [df_lab, df_proc, df_dx]
 #           df = pd.concat(frames)
 #       else:
 #           frames = [df, df_lab, df_proc, df_dx]
 #           df = pd.concat(frames)
 #       count +=1
        
        
        #if query[1]>0:
        #    dfs.append(df_lab, df_proc, df_dx)
            
        #else: 
        #    if query[0][2]<0:
        #        dfs.append(df_lab, df_proc, df_dx)
        #    else:
        #        dfs.append(df_lab, df_proc, df_dx)
    #Make the dataframe into SQL table:
    from pandas.io import sql
    c.execute('DROP TABLE IF EXISTS CHF_dataframe')
    sql.to_sql(df, name = 'CHF_dataframe', con=conn, index=False, index_label = 'ROW_ID', if_exists = 'append')
    
    c.close()
    conn.close()
    
    print ("UFM TABLES COMPLETE!")
    
    return (df, subj, hadm, s, h)


## This is the actual construction of the FEATURE tables from UFM tables. #####
### Will likely use time series data. #########################################

##need this auxiliary function##
def rep_digit(s):
    try: 
        int(s)
        return (True)   
    except: 
        return (False)
        
def init_features(ufm):
    #make empty features table
    lab_f = list(set(ufm[ufm['TYPE']=='l']['FEATURE']))
    proc_f = list(set(ufm[ufm['TYPE']=='p']['FEATURE']))
    dx_f = list(set(ufm[ufm['TYPE']=='d']['FEATURE']))
    dx_f=list(filter(None,dx_f))
    
    columns = lab_f+proc_f+dx_f
    columns = list(set([str(i) for i in columns]))
    print (len(columns))
    features = pd.DataFrame(columns=columns)
    
    #fill missing lab and proc values:
    #1. create dictionary of mean lab and proc values
    mean_values = {i:np.mean(np.array([int (j) for j in list(ufm.loc[ufm['FEATURE']==i, 'VALUE']) if rep_digit(j)])) for i in proc_f+lab_f}    
    mean_values = {str(i):j for i,j in mean_values.items()}
    #PICKLE THIS.
    #2. iterate through list, append to dictionary for non-redundant values.
    for i in columns:
        if i not in mean_values.keys():
            mean_values[i]=0.0
    for key in mean_values.keys():
        features.loc[0, key] = mean_values[key]
    print (len(features))           #now a (1,4122) dataframe
    
    return (features)
        


def get_seq(df, features):
    X = []
    first_time = 0
    time=0
    for index, row in df.iterrows():
        if first_time == 0:
            t_prev = row['TIME']
            if (rep_digit(row['VALUE'])):
                features.loc[time, str(row['FEATURE'])] = row['VALUE']
            first_time+=1
        else:
            t_curr = row['TIME']
            if ((t_prev ==t_curr) & rep_digit(row['VALUE'])):
                features.loc[time, str(row['FEATURE'])] = row['VALUE']
            else:
                t_prev = row['TIME']
                time+=1
                features.loc[time] = features.loc[time-1]
                if (rep_digit(row['VALUE'])):
                    features.loc[time, str(row['FEATURE'])] = row['VALUE']
    
    features=features.fillna(0)
    X = features.as_matrix()
    X= X.astype(dtype='float32')
            
        
    return (X)

def embedding (ufm, t_queries):
    #tallies = ufm.groupby('TYPE').FEATURE.nunique()
    
    #convert 'Time' in UFM table to timestamp for filtering.
    ufm['TIME'] = pd.to_datetime(ufm['TIME'])
    
    print ('\n' + 'Making X sequences for HMM for each CHF related encounter.')
    print ('This may take a while...')
    print ('+++++++++++IN PROCESS++++++++++++++')
    
    #sequences of observations
    X=[]
    lengths = []
    
    training_set = t_queries[0:7000]
    #separate into two sets for the two HMMs
    pos_set = [i for i in training_set if i[3] == 1]
    neg_set = [i for i in training_set if i[3] == 0]
    cross_validation = t_queries[7000:9000]
    test_set = t_queries[9000:]
    
    #get all the pos_set arrays to feed into HMM1
    sess = 0
    for query in pos_set:
        sess+=1
        #very important: REINITIALIZE FEATURES everytime
        try:
            with open ("/media/sf_mimic/csv/CHF ANALYSIS/vars/init_features.pickle", "rb") as f:
                features = pickle.load(f)
        except:
                features = init_features(ufm)
        
        #initialize parameters from training set
        s, t1, t2 = query[0], query[1], query[2]
        mask = (ufm['TIME'] >= t1) & (ufm['TIME'] <=t2)
        #df = ufm.loc[mask]
        #df = df[df['SUBJECT_ID'] == s]
        df = (ufm.loc[mask])[(ufm.loc[mask])['SUBJECT_ID']==s]
        df = df.sort('TIME', ascending = True)
        
        print ("Currently on Session: {0} out of {1}.".format(sess, len(pos_set)))   
        print ("DF size: {0}, features size: {1}".format(len(df), len(features)))
        
        x= get_seq(df, features)
        print ("Size of x: {0}".format(x.shape))
        print ("... should have 4120: {0}".format(x.shape[1]))
        #X.append(x): memmap style
        b = np.memmap(file_b, mode = 'w+', shape = (x.shape[0], x.shape[1]), dtype = 'float32')
        b[:] = x
        
        #make buffer file from file_a to concatenate with file_b
        if not lengths: 
            prev =0
            a = np.memmap(file_a, mode = 'w+', shape = (x.shape[0], x.shape[1]), dtype='float32')
            a[:] = b[:]
        else:
            #a=np.memmap(file_a, mode = 'r+', shape = (prev, x.shape[1]), dtype='float32')
            X = np.memmap(file_a, mode = 'r+', shape = (prev+len(x), x.shape[1]), dtype='float32')
            #X[:prev, :] = a
            X[prev:, : ] = b
            print ("Memory of X: {0}".format(sys.getsizeof(X)))
        lengths.append(len(x))
        prev = sum(lengths)
        print ("Total rows in X: {0}".format(prev))
    
    
    print ("X COMPLETE.")
    pickle_out = open("/media/sf_mimic/csv/CHF ANALYSIS/vars/X_pos.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle.close()
    print ("... and Pickled.")

    del b
    
    #Pass to HMM:
    #model = GaussianHMM(n_components=4).fit(X)
    #model.score(x)
    
    return (X, lengths, training_set, cross_validation, test_set, pos_set)
        


    #HMM: O = np.column_stack([pts, ast, rbs])

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    desc = "Welcome to CHF Readmission Predictor by af1tang."
    version = "version 0.1"
    opt = OptionParser (description = desc, version=version)
    opt.add_option ('-i', action = 'store', type ='string', dest='input', help='Please input path to Database File.')
    opt.add_option ('-o', action = 'store', type = 'string', dest='output', default='CHF_data.pickle', help='Please state desired storage file for this session.')
    (cli, args) = opt.parse_args()
    opt.print_help()
    
    #df, admissions, patients, subjects, admits, y_positive, y_negative = feature_table() 
    
    #with open ("/media/sf_mimic/csv/CHF ANALYSIS/vars/flags.pickle","rb") as f:
    #    flags= pickle.load(f)
    #with open ("/media/sf_mimic/csv/CHF ANALYSIS/vars/y_negative.pickle","rb") as f:
    #    y_negative=pickle.load(f)
    #with open ("/media/sf_mimic/csv/CHF ANALYSIS/vars/y_positive.pickle","rb") as f:
    #    y_positive=pickle.load(f) 
    #with open ("/media/sf_mimic/csv/CHF ANALYSIS/vars/t_pos.pickle","rb") as f:
    #    t_pos=pickle.load(f) 
    #with open ("/media/sf_mimic/csv/CHF ANALYSIS/vars/t_neg.pickle","rb") as f:
    #    t_neg=pickle.load(f) 
    with open ("/media/sf_mimic/csv/CHF ANALYSIS/vars/t_queries.pickle","rb") as f:
        t_queries=pickle.load(f) 
    
    with open ("/media/sf_mimic/csv/CHF ANALYSIS/vars/ufm.pickle","rb") as f:
        ufm=pickle.load(f)
    
    #with open ("/media/sf_mimic/csv/CHF ANALYSIS/vars/mean_values2.pickle", "rb") as f:
    #    mean_values = pickle.load(f)
    
    #df, subj, hadm, s, h = x_features(y_positive, y_negative, flags, t_pos, t_neg)
    X, lengths, training_set, cross_validation, test_set, pos_set= embedding (ufm, t_queries)    
    #tf.app.run()
    
