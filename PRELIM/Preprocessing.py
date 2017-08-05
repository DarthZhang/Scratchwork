# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:50:50 2017

@author: af1tang
"""

import pandas as pd
import numpy as np
import pickle
import gzip
import math
import datetime, time
from itertools import tee
from pandas.tools.plotting import scatter_matrix
from scipy import stats
#import pymysql as mysql

filename_dx = ''
filename_pts = ''
filename_admits = ''
filename_icustays = ''
filename_labevents = ''
filename_chartevents = ''
filename_ddx = ''

labs = {50862:0, 50885:1, 51006:2, 50893:3, 50912:4, 50983: 5, 50971: 6, 50882: 7, 50931: 8, 50820: 9, 50816:10, 50817: 11, 50818: 12, 51265: 13, 51108: 14, 51300:15, 50960:16, 50813:17 }
#full_labs = {50862:0, 51006:1, 50893:2, 50912:3, 50983: 4, 50971: 5, 50882: 6, 50931: 7, 50820: 8, 50818: 9, 51265: 10, 50960:11, 50813:12}
full_labs = {50862:6, 51006:7, 50893:8, 50912:9, 50983: 10, 50971: 11, 50882: 12, 50931: 13, 50820: 14, 50818: 15, 51265: 16, 50960:17, 50813:18}

#chartevents1 = {221: 0, 220045: 0, 3313: 1, 3315: 1, 3317:1, 3319: 1, 3321: 1, 3323: 1, 3325:1, 8502: 2, 8503: 2, 8504: 2, 8505: 2, 8506: 2, 8507: 2, 8508:2, 198: 3, 676: 4, 677:4, 223762:4, 678:4, 679:4, 7884:5, 3603:5, 8113: 5, 618:5, 220210: 5, 227428:6}

#chartevents2 = {221: 0, 220045:0, 6: 1, 455:1, 51:1, 442:1, 6701:1, 220179:1, 220050:1, 8364:2, 8441:2, 8368:2, 8440:2, 8555:2, 220180:2, 220051:2, 223761:3, 678:3, 223762:4, 676:4, 7884:5, 3603:5, 8113:5, 618:5, 615:5, 220210:5, 224690:5, 646:6, 220277:6, 227428:7}
chartevents = {221: 0, 220045:0, 6: 1, 455:1, 51:1, 442:1, 6701:1, 220179:1, 220050:1, 8364:2, 8441:2, 8368:2, 8440:2, 8555:2, 220180:2, 220051:2, 223761:3, 678:3, 679: 3, 7884:4, 3603:4, 8113:4, 618:4, 615:4, 220210:4, 224690:4, 646:5, 220277:5} 
lab_names = {50813: 'Lactate', 50818: 'PaCO2', 50820: 'PH', 50862: 'Albumin', 50882: 'HCO3', 50893: 'Ca', 50912: 'Cre', 50931: 'Glc', 50960: 'Mg', 50971: 'K', 50983: 'Na', 51006: 'BUN', 51265: 'Platelets'}
chart_names = {0: 'HR', 1: 'SBP', 2: 'DBP', 3: 'TEMP', 4: 'RR', 5: 'SPO2'}

#mimic = 'MIMIC3'
#host = 'illidan-gpu-1.egr.msu.edu'
#user = 'af1tang'
#pw = 'illidan'    
#port = 3306

#connect to MySQL using engine to write pandas df --> mysql
#conn = mysql.connect(host = host, user = user, passwd = pw, db = mimic, port = port)    
#engine = create_engine ("mysql+pymysql://af1tang:illidan@illidan-gpu-1.egr.msu.edu:3306/MIMIC3")

def wrangling (stays, diagnoses):
    #1. only adult patients with LOS > 24h
    cohort = stays[(stays.AGE>= 18) & (stays.LOS>=1.0)]
    
    #create a 30d and 1 yr window
    cohort ['30D'] = cohort.DISCHTIME + datetime.timedelta(days=30)
    cohort ['1YR'] = cohort.DISCHTIME + datetime.timedelta(days=365)
    
    #update mortality @ 30d and 1yr
    mortality = cohort.DOD.notnull() & ((cohort.INTIME <= cohort.DOD) & (cohort['30D'] >= cohort['DOD']))
    mortality = mortality | (cohort.DEATHTIME.notnull() & ((cohort.INTIME <= cohort.DEATHTIME) & (cohort['30D'] >= cohort['DEATHTIME'])))
    cohort['MORTALITY_30D'] = mortality.astype(int)
    mortality = cohort.DOD.notnull() & ((cohort.INTIME <= cohort.DOD) & (cohort['1YR'] >= cohort['DOD']))
    mortality = mortality | (cohort.DEATHTIME.notnull() & ((cohort.INTIME <= cohort.DEATHTIME) & (cohort['1YR'] >= cohort['DEATHTIME'])))
    cohort['MORTALITY_1YR'] = mortality.astype(int)
    mortality = cohort.DOD.notnull() | cohort.DEATHTIME.notnull()
    cohort['MORTALITY'] = mortality.astype(int)
    
    #merge with diagnoses to make 
    diagnoses = cohort.merge(diagnoses, on = ['SUBJECT_ID', 'HADM_ID'])
    diagnoses = diagnoses[diagnoses.HAS_CHARTEVENTS_DATA==1]
    
    #count ICD9 "phenotypes"
    codes = diagnoses[['ICD9_CODE','LONG_TITLE']].drop_duplicates().set_index('ICD9_CODE')
    codes['COUNT'] = diagnoses.groupby('ICD9_CODE')['ICUSTAY_ID'].count()
    codes.COUNT = codes.COUNT.fillna(0).astype(int)
    codes = codes.ix[codes.COUNT>0]
    
    #readmit table
    df = cohort[cohort.MORTALITY ==0] [['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME']].drop_duplicates()
    #df.set_index('SUBJECT_ID', inplace = True)
    readm = df.groupby('SUBJECT_ID').filter(lambda x: len(x['HADM_ID'])>1)
    readm = readm.sort(['SUBJECT_ID', 'ADMITTIME'], ascending = [True, False])
    #readm = readm.set_index('SUBJECT_ID')
    readm['diff'] = np.nan
    for s in list(set(readm.SUBJECT_ID)):
        readm.ix[readm.SUBJECT_ID == s, 'diff'] = readm[readm.SUBJECT_ID == s].DISCHTIME - readm[readm.SUBJECT_ID==s].ADMITTIME.shift(1)
    readm['diff'] = readm['diff'].apply(lambda x: -1.0* x.days if str(x) != 'NaT' else 0)
    readm = readm[readm['diff'] != 0]
    
    #4. preliminary pos-neg split
    pos, neg = pos_neg_split(stays, admits)    
    hadm = [s[2] for s in pos+neg]
    
    labevents = lab_features(hadm, labs)
    
    return (pos, neg)

def lab_features(hadm, labs):
    labevents = {}
    chunksize = 100000
    vocab = list(labs.keys())
    for h in hadm:
        labevents[h] = [None]*len(vocab)

    with gzip.open(filename_labevents, 'r') as f:
        for numlines, l in enumerate(f): pass
    numlines+=1
    cols = list(pd.read_csv(filename_labevents, nrows=0).columns)
    for i in range(0, numlines, chunksize):
        print(i)
        df = pd.read_csv(filename_labevents, header = None, nrows = chunksize, skiprows= i, names = cols)
        temp = df[(df.HADM_ID.isin(hadm)) & (df.ITEMID.isin(vocab))]
        for h in list(set(temp.HADM_ID)):
            mins = temp[temp.HADM_ID==h].groupby(['ITEMID']).min().VALUENUM
            maxs = temp[temp.HADM_ID==h].groupby(['ITEMID']).max().VALUENUM
            sums = temp[temp.HADM_ID==h].groupby(['ITEMID']).sum().VALUENUM
            counts = temp[temp.HADM_ID==h].groupby(['ITEMID']).size()
            h = int(h)
            for item in counts.index:
                if labevents[h][labs[item]] == None:
                    mean = 1.0*sums[item] / counts[item]
                    labevents[h][labs[item]] = [mins[item], maxs[item], mean, counts[item]]
                else:
                    if mins[item] < labevents[h][labs[item]][0]:
                        labevents[h][labs[item]][0] = mins[item]
                    if maxs[item] > labevents[h][labs[item]][1]:
                        labevents[h][labs[item]][1] = maxs[item]                           
                    mean = (labevents[h][labs[item]][2] * labevents[h][labs[item]][3] + sums[item])/(counts[item] + labevents[h][labs[item]][3])
                    labevents[h][labs[item]][2] = mean
                    labevents[h][labs[item]][3] += counts[item]
    
    with open ('/home/andy/Desktop/MIMIC/vars/labs.pkl', 'wb') as f:
        pickle.dump(labevents, f)
    
    summary = []; 
    for i in range(len(vocab)):
        summary.append([99999, 0, 0, 0])
    
    for h in hadm:
        for item in labevents[h]:
            if item == None:
                pass
            else:
                if item[0] < summary[labevents[h].index(item)][0]: 
                    summary[labevents[h].index(item)][0] = item[0]
                if item[1] > summary[labevents[h].index(item)][1]:
                    summary[labevents[h].index(item)][1] = item[1]
                if math.isnan(float(item[2])): 
                    print (item[2])
                else:
                    summary[labevents[h].index(item)][2] += (item[2] * item[3])
                    summary[labevents[h].index(item)][3] += item[3]
    for i in range(len(summary)):
        mean = 1.0*summary[i][2]/summary[i][3]
        summary[i][2] = mean
        
    return (labevents)
    
def chart_features(hadm, chartevents):
    vitals = {}
    chunksize = 10**6
    vocab = list(chartevents.keys())
    
    for h in hadm:
        vitals[h] = [None]*len(list(set(chartevents.values())))
        
    with gzip.open(filename_chartevents, 'r') as f:
        for numlines, l in enumerate(f): pass
    numlines+=1
    
    cols = list(pd.read_csv(filename_chartevents, nrows=0).columns)
    for i in range(0, numlines, chunksize):
        print(i)
        df = pd.read_csv(filename_chartevents, header = None, nrows = chunksize, skiprows= i, names = cols)
        temp = df[(df.HADM_ID.isin(hadm)) & (df.ITEMID.isin(vocab))]
        for h in list(set(temp.HADM_ID)):
            #print (h)
            mins = temp[(temp.HADM_ID==h) & (temp.ERROR==0)].groupby(['ITEMID']).min().VALUENUM
            maxs = temp[(temp.HADM_ID==h) & (temp.ERROR==0)].groupby(['ITEMID']).max().VALUENUM
            sums = temp[(temp.HADM_ID==h) & (temp.ERROR==0)].groupby(['ITEMID']).sum().VALUENUM
            #means = temp[(temp.HADM_ID==h) & (temp.ERROR==0)].groupby(['ITEMID']).mean().VALUENUM
            counts = temp[(temp.HADM_ID==h) & (temp.ERROR==0)].groupby(['ITEMID']).size()
            for item in counts.index:
                if vitals[h][chartevents[item]] == None:
                    mean = 1.0*sums[item] / counts[item]
                    vitals[h][chartevents[item]] = [mins[item], maxs[item], mean, counts[item]]
                else:
                    if mins[item] < vitals[h][chartevents[item]][0]:
                        vitals[h][chartevents[item]][0] = mins[item]
                    if maxs[item] > vitals[h][chartevents[item]][1]:
                        vitals[h][chartevents[item]][1] = maxs[item]
                    mean = (vitals[h][chartevents[item]][2] * vitals[h][chartevents[item]][3] + sums[item])/(counts[item] + vitals[h][chartevents[item]][3])
                    vitals[h][chartevents[item]][2] = mean
                    vitals[h][chartevents[item]][3] += counts[item]
    with open ('/home/andy/Desktop/MIMIC/vars/charts/charts.pkl', 'wb') as f:
        pickle.dump(vitals, f)
    summary = []; 
    for i in range(len(list(set(chartevents.values())))):
        summary.append([99999, 0, 0, 0])
    
    for h in hadm:
        for item in vitals[h]:
            if item == None:
                pass
            else:
                if item[0] < summary[vitals[h].index(item)][0]: 
                    summary[vitals[h].index(item)][0] = item[0]
                if item[1] > summary[vitals[h].index(item)][1]:
                    summary[vitals[h].index(item)][1] = item[1]
                if math.isnan(float(item[2])): 
                    print (item[2])
                else:
                    summary[vitals[h].index(item)][2] += (item[2] * item[3])
                    summary[vitals[h].index(item)][3] += item[3]
    for i in range(len(summary)):
        mean = 1.0*summary[i][2]/summary[i][3]
        summary[i][2] = mean
    
    return (vitals)

def get_stats(events, dct):
    data = pd.DataFrame(events)
    data = data.transpose()
    reverse_dct = {v: k for k, v in dct.items()}
    cols = list(reverse_dct.values())
    
    df = data.dropna(axis=0, threshold = len(cols))
    df = df.apply(pd.to_numeric, errors = 'ignore')
    df.columns = cols
    df.index.names = ['HADM_ID']

    #get rid of outliers
    #df = df[np.abs(df.HR-df.HR.mean())<=(3*df.HR.std())]    
    
    hadm = list(set(df.index))
    #plotting
    #describe = df.describe(); describe.to_csv(outfile_labs_describe)
    #df.hist()
    #scatter_matrix(df, alpha = .2, figsize =(13,13), diagonal = 'kde')
    quints = {}
    for c in cols:
        cuts = pd.qcut(df[c], 5, retbins = True)
        #idx = reverse_names[c]
        quints [c] = cuts[1]
    
    return (quints, hadm)
    
def sentences (hadm, dictionary, filename):
    charts = {}; #l_total = {};
    chunksize = 100000
    vocab = list(dictionary.keys())
    for h in hadm:
        charts[h] = {}
    #for k in vocab:
     #   l_total[k] = []

    cols = list(pd.read_csv(filename, nrows=0).columns)
    
    #### OPTIONAL ####
    #Get ranking of Chartevents by sampling frequency#
    items= pd.read_csvitems = pd.read_csv('/home/andy/Desktop/MIMIC/csv/D_ITEMS.csv.gz' )
    events = {}
    for l in list(set(items.ITEMID)):
        events[l] = 0
    
    start = time.time(); count = 0
    for df in pd.read_csv(filename, iterator = True, chunksize = chunksize):
        count+=1
        print ("Chunk: {0}, {1}".format(df.shape[0]*count, time.time() - start))
        
        for idx in list(set(df.groupby('ITEMID').size().index)):
            events[idx] += df.groupby('ITEMID').size()[idx]
    sorted_events = sorted(events.items(), key = lambda x: x[1], reverse = True)
    lst = []
    for i in sorted_events:
        lst.append([i[0], items[items.ITEMID==i[0]].LABEL.values[0], i[1]])
    
    ####Make Sentences####    
    start = time.time(); count = 0
    for df in pd.read_csv(filename, iterator = True, chunksize = chunksize):
        count+=1
        print ("Chunk: {0}, {1}".format(df.shape[0]*count, time.time() - start))

        df.columns = cols
        df = df[(df.ERROR ==0) & (df.HADM_ID.isin(hadm)) & (df.ITEMID.isin(vocab))][['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM']]
        df['TIME'] = df['CHARTTIME'].values.astype('<M8[h]')
        
        temp = pd.DataFrame({'sum': df.groupby(['HADM_ID','TIME', 'ITEMID'])['VALUENUM'].sum(), 'count': df.groupby(['HADM_ID', 'TIME', 'ITEMID'])['VALUENUM'].size()}).reset_index()
        dct = temp.to_dict(orient='split')['data']
        for d in dct:
            if d[1] not in charts[d[0]].keys():
                charts[d[0]][d[1]] = {dictionary[d[2]]: [d[3], d[4]]}
            elif d[2] not in charts[d[0]][d[1]].keys():
                charts[d[0]][d[1]][dictionary[d[2]]] = [d[3], d[4]]
            else:
                charts[d[0]][d[1]][dictionary[d[2]]][0] += d[3]
                charts[d[0]][d[1]][dictionary[d[2]]][1] += d[4]
        del dct; del temp
    
    events = {}
    for h in charts.keys():
        events[h] = {}
        for t in charts[h].keys():
            events[h][t] = [None]*19
            for k, v in charts[h][t].items():
                events[h][t][k] = v[1]/v[0]
    events = {}
    for h in charts.keys():
        events[h] = []
        for t in charts[h].keys():
            temp = [None]*19
            for k, v in charts[h][t].items():
                temp[k] = v[1]/v[0]
            events[h].append([t, temp])    
            
        #temp = list(df[df.ITEMID.isin(vocab)][['ITEMID', 'VALUENUM']].values.tolist())
        #for item in temp:
        #    l_total[item[0]].append(item[1])
                      
        #make sentences for target patients
        temp = df[(df.HADM_ID.isin(hadm)) & (df.ITEMID.isin(vocab))]
        admissions = list(set(temp.HADM_ID.dropna()))
        for a in admissions:
            tmp = sorted(list(temp[temp.HADM_ID==a][['ITEMID', 'VALUENUM', 'CHARTTIME']].values.tolist()), key = lambda x: x[2])
            for item in tmp:
                q = quints[labs[item[0]]]
                if item[1] <= q[1]: string = '_1'
                elif q[1] < item[1] <= q[2]: string = '_2'
                elif q[2]<item[1] <=q[3]: string = '_3'
                elif q[3] <item[1]<=q[4]: string = '_4'
                elif item[1] > q[4]: string = '_5'
                else: print(item[1])
                sentences[a].append([str(item[0]) + string, item[2]])
                #if item[1]>=0: l_sentences[a].append(item)
    
    counts = []
    for h in hadm: 
        counts.append(len(sentences[h]))
    print ("average sentence length: {0}".format(np.mean(counts)))  #216.4 events per hadm
    
    return (sentences)
    
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
    
def pos_neg_split(subj, admits, HADM):
    count = 0
    neg = []; pos = []
    for s in subj:
        plus = []
        minus=[]
        count+=1; print (count)
        hadm = list(set(admits[admits['SUBJECT_ID']==s].HADM_ID.values))
        if len(hadm) <1: 
            pass
        else:   
            t = [(pd.to_datetime(admits[admits['HADM_ID']==i]['ADMITTIME'].values[0]), i) for i in hadm]
            #dt = [(pd.to_datetime(admits[admits['HADM_ID']==i]['DISCHTIME'].values[0]), i) for i in hadm]
            t = sorted(t, reverse = True)
            dt = {}
            for i in hadm:
                dt[i] = pd.to_datetime(admits[admits['HADM_ID']==i]['DISCHTIME'].values[0])
            for t2,t1 in pairwise(iterable = t):
                if (t2[0]-dt[t1[1]]).days<=30:
                    plus.append((s, t1[1], 1))
                elif (t1[1] in HADM):
                    minus.append((s, t1[1], 0))
            #    if (t2[0] - t1[0]).days >30:
            #        minus.append((s, t1[0], t1[1], 0))
            #    else:
            #        plus.append((s, t1[0], t1[1], 1))
            if len(plus) >0:
                for k in plus:
                    if k[1] in HADM:
                        pos.append(plus[0])
            elif len(minus)>0:
                if t[0][1] in HADM:
                    neg.append((s, t[0][1], 0))
                else:
                    neg.append(minus[0])
            elif len(hadm)<2:
                #if hadm[0] in HADM:
                #    neg.append((s, hadm[0], 0))
                pass
    return (pos, neg)

def demographics (subj, HADM):
    demo = {}    
    
    marital = {'MARRIED': 0, 'SEPARATED': 1, 'SINGLE':2, 'WIDOWED':3, 'DIVORCED': 4, 'LIFE PARTNER':5, 'UNKNOWN (DEFAULT)': 6}
    ethn = {'OTHER': 0, 'WHITE': 1, 'CARIB': 2, 'ASIAN': 3, 'AMERI': 4, 'BLACK': 5, 'HISPA': 6, 'SOUTH': 7, 'MULTI': 8, 'MIDDL': 9, 'PORTU': 10, 'NATIV': 11, 'UNABL': 12}
    insurance = {'Medicaid': 0, 'Government': 1, 'Medicare': 2, 'Self Pay': 3, 'Private': 4}
    
    for s in subj:
        h = admits[(admits.SUBJECT_ID==s) & (admits.HADM_ID.isin(HADM))].HADM_ID.values[0]
        t = pd.to_datetime(admits[(admits.SUBJECT_ID==s) & (admits.HADM_ID==h)].ADMITTIME.values[0])
        dob = pd.to_datetime(pts[pts.SUBJECT_ID==s].DOB.values[0])
        #age
        age = round(((t-dob).days)/365)
        if age >100:
            age = 89.0
        #LOS
        los =  icustays[icustays.SUBJECT_ID == s].LOS.values[0]
        #insurance
        insure = admits[admits.HADM_ID ==h].INSURANCE.values[0]
        
        demo [s] = {'age': age, 'LOS': los, 'insurance': insurance[insure]}
        
    return (demo)

def group_icd9 (subj, HADM):
    count = 0;
    with open ('/home/andy/Desktop/MIMIC/vars/dx/feature_dictionary.pkl', 'rb') as f:
        dct = pickle.load(f)
    dx = {}
    d_sentences = {}
    for s in subj:
        dx[s] = [0] * len(dct)
        
        count+=1; print (count)
        hadm = list(set(admits[admits['SUBJECT_ID']==s].HADM_ID.values))        
        h = admits[(admits.SUBJECT_ID==s) & (admits.HADM_ID.isin(HADM))].HADM_ID.values[0]
        
        t = [(pd.to_datetime(admits[admits['HADM_ID']==i]['ADMITTIME'].values[0]), i) for i in hadm]
        t0 = pd.to_datetime(admits[admits.HADM_ID==h].ADMITTIME.values[0])
        
        d = []
        for i in t:
            if i[0] <= t0:
                temp = list(set(diagnoses[diagnoses.HADM_ID==h].ICD9_CODE))
                temp = [x for x in temp if str(x) != 'nan']
                d.append([ch[0:3] for ch in temp])
                for j in temp:
                    dx[s][dct[j[0:3]]] += 1        
        d_sentences[s] = d
                
    return (dx, d_sentences)

if __name__ == '__main__':
    diagnoses = pd.read_csv(filename_dx)
    admits = pd.read_csv(filename_admits)
    pts = pd.read_csv(filename_pts)
    icustays = pd.read_csv(filename_icustays)
    ddx = pd.read_csv(filename_ddx)
    del icustays['ROW_ID'], admits['ROW_ID'], pts['ROW_ID'], diagnoses['ROW_ID'], ddx['ROW_ID'], diagnoses['SEQ_NUM']
    
    #make uber patient admissions chart
    stays = icustays.merge(pts, on = 'SUBJECT_ID')
    stays = stays.merge(admits, on = ['SUBJECT_ID', 'HADM_ID'])
    
    #remove ICUstays with transfers
    stays = stays.ix[(stays.FIRST_WARDID == stays.LAST_WARDID) & (stays.FIRST_CAREUNIT == stays.LAST_CAREUNIT)]
    
    #to datetime
    stays['INTIME'] = pd.to_datetime(stays['INTIME'])
    stays['OUTTIME'] = pd.to_datetime(stays['OUTTIME'])
    stays['DOB'] = pd.to_datetime(stays['DOB'])
    stays['DOD'] = pd.to_datetime(stays['DOD'])
    stays['DOD_HOSP'] = pd.to_datetime(stays['DOD_HOSP'])
    stays['DOD_SSN'] = pd.to_datetime(stays['DOD_SSN'])
    stays['ADMITTIME'] = pd.to_datetime(stays['ADMITTIME'])
    stays['DISCHTIME'] = pd.to_datetime(stays['DISCHTIME'])
    stays['DEATHTIME'] = pd.to_datetime(stays['DEATHTIME'])
    stays['EDREGTIME'] = pd.to_datetime(stays['EDREGTIME'])
    stays['EDOUTTIME'] = pd.to_datetime(stays['EDOUTTIME'])
    
    #add age
    stays['AGE'] = (stays.INTIME - stays.DOB).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60/24/365
    stays.AGE.ix[stays.AGE<0] = 90
    
    #mortality calculations
    #in hospital
    mortality = stays.DOD.notnull() & ((stays.ADMITTIME <= stays.DOD) & (stays.DISCHTIME >= stays.DOD))
    mortality = mortality | (stays.DEATHTIME.notnull() & ((stays.ADMITTIME <= stays.DEATHTIME) & (stays.DISCHTIME >= stays.DEATHTIME)))
    stays['MORTALITY_INHOSPITAL'] = mortality.astype(int)
    
    #in ICU
    mortality = stays.DOD.notnull() & ((stays.INTIME <= stays.DOD) & (stays.OUTTIME >= stays.DOD))
    mortality = mortality | (stays.DEATHTIME.notnull() & ((stays.INTIME <= stays.DEATHTIME) & (stays.OUTTIME >= stays.DEATHTIME)))
    stays['MORTALITY_INUNIT'] = mortality.astype(int)
    
    #merge ddx with diagnoses 
    diagnoses = diagnoses.merge(ddx, on = 'ICD9_CODE')
    
'''
SCRATCH WORK
    #2. age > 18 or age >=65    
    #make a dictionary for patient age
    ages = {}; adults =[]; mature = []
    for s in list(set(alive.SUBJECT_ID)):
        hadm = list(set(admits[admits.SUBJECT_ID==s].HADM_ID.values))
        t = [(pd.to_datetime(admits[admits['HADM_ID']==i].ADMITTIME.values[0]), i) for i in hadm]
        t = sorted(t, reverse = True); t1 = t[0][0]
        dob = pd.to_datetime(alive[alive.SUBJECT_ID==s].DOB.values[0])
        age = round(((t1-dob).days)/365)
        ages[s] = age
        if age > 15: adults.append(s)
        if age >= 65: mature.append(s)
    select = admits[admits.SUBJECT_ID.isin(adults)]
    dsch = set(admits.DISCHARGE_LOCATION); lst = list(dsch.symmetric_difference(['DEAD/EXPIRED', 'LEFT AGAINST MEDICAL ADVI', 'SHORT TERM HOSPITAL']))
    adults = list(set(select[select.DISCHARGE_LOCATION.isin(lst)].SUBJECT_ID))
    
    #3. ICU >24hrs
    stays = list(set(icustays[(icustays.SUBJECT_ID.isin(adults)) & (icustays.LOS>=1.0)].SUBJECT_ID))

'''
