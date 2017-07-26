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

labs = {50862:0, 50885:1, 51006:2, 50893:3, 50912:4, 50983: 5, 50971: 6, 50882: 7, 50931: 8, 50820: 9, 50816:10, 50817: 11, 50818: 12, 51265: 13, 51108: 14, 51300:15, 50960:16, 50813:17 }
full_labs = {50862:0, 51006:1, 50893:2, 50912:3, 50983: 4, 50971: 5, 50882: 6, 50931: 7, 50820: 8, 50818: 9, 51265: 10, 50960:11, 50813:12}
chartevents = {221: 0, 220045: 0, 3313: 1, 3315: 1, 3317:1, 3319: 1, 3321: 1, 3323: 1, 3325:1, 8502: 2, 8503: 2, 8504: 2, 8505: 2, 8506: 2, 8507: 2, 8508:2, 198: 3, 676: 4, 677:4, 223762:4, 678:4, 679:4, 7884:5, 3603:5, 8113: 5, 618:5, 220210: 5, 227428:6}

lab_names = {50813: 'Lactate', 50818: 'PaCO2', 50820: 'PH', 50862: 'Albumin', 50882: 'HCO3', 50893: 'Ca', 50912: 'Cre', 50931: 'Glc', 50960: 'Mg', 50971: 'K', 50983: 'Na', 51006: 'BUN', 51265: 'Platelets'}

#mimic = 'MIMIC3'
#host = 'illidan-gpu-1.egr.msu.edu'
#user = 'af1tang'
#pw = 'illidan'    
#port = 3306

#connect to MySQL using engine to write pandas df --> mysql
#conn = mysql.connect(host = host, user = user, passwd = pw, db = mimic, port = port)    
#engine = create_engine ("mysql+pymysql://af1tang:illidan@illidan-gpu-1.egr.msu.edu:3306/MIMIC3")

def wrangling (dx, admits, pts):
    #1. only alive aptients
    alive = pts[pts.EXPIRE_FLAG==0]
    dead = list(set(pts[pts.EXPIRE_FLAG==1].SUBJECT_ID))
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
    chunksize = 100000
    vocab = list(chartevents.keys())
    
    for h in hadm:
        vitals[h] = [None]*len(vocab)

        
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
    with open ('/home/andy/Desktop/MIMIC/vars/charts.pkl', 'wb') as f:
        pickle.dump(vitals, f)
    charts = {}
    for h in hadm:
        charts[h] = [[c[0], c[1], c[2]] for c in vitals[h]]
    
    return (charts)

def get_stats(events, dct):
    data = pd.DataFrame(events)
    data = data.transpose()
    reverse_dct = {v: k for k, v in dct.items()}
    cols = list(reverse_dct.values())
    
    df = data.dropna(axis=0, how = 'any')
    df = df.apply(pd.to_numeric, errors = 'ignore')
    df.columns = cols
    df.index.names = ['HADM_ID']
    
    hadm = list(set(df.HADM_ID))
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
    
    
def l_sentences (hadm, labs, quints):
    l_sentences = {}; #l_total = {};
    chunksize = 100000
    vocab = list(labs.keys())
    for h in hadm:
        l_sentences[h] = []
    #for k in vocab:
     #   l_total[k] = []

    with gzip.open(filename_labevents, 'r') as f:
        for numlines, l in enumerate(f): pass
    numlines+=1
    
    cols = list(pd.read_csv(filename_labevents, nrows=0).columns)
    for i in range(0, numlines, chunksize):
        print (i)
        #add rows to l_total
        df = pd.read_csv(filename_labevents, header = None, nrows = chunksize, skiprows= i, names = cols)
        #temp = list(df[df.ITEMID.isin(vocab)][['ITEMID', 'VALUENUM']].values.tolist())
        #for item in temp:
        #    l_total[item[0]].append(item[1])
               
        #make sentences for target patients
        temp = df[(df.HADM_ID.isin(hadm)) & (df.ITEMID.isin(vocab))]
        admissions = list(set(temp.HADM_ID.dropna()))
        for a in admissions:
            tmp = sorted(list(temp[temp.HADM_ID==a][['ITEMID', 'VALUENUM', 'CHARTTIME']].values.tolist()), key = lambda x: x[2])
            for item in tmp:
                q = quints[item[0]]
                if item[1] <= q[1]: string = '_1'
                elif q[1] < item[1] <= q[2]: string = '_2'
                elif q[2]<item[1] <=q[3]: string = '_3'
                elif q[3] <item[1]<=q[4]: string = '_4'
                elif item[1] > q[4]: string = '_5'
                else: print(item[1])
                l_sentences[a].append([str(item[0]) + string, item[2]])
                #if item[1]>=0: l_sentences[a].append(item)
    
    counts = []
    for h in hadm: 
        counts.append(len(l_sentences[h]))
    print ("average lab sentence length: {0}".format(np.mean(counts)))  #216.4 events per hadm
    
    return (l_sentences)
    
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
                pos.append(plus[0])
            elif len(minus)>0:
                if t[0][1] in HADM:
                    neg.append((s, t[0][1], 0))
                else:
                    neg.append(minus[0])
            elif len(hadm)<2:
                if hadm[0] in HADM:
                    neg.append((s, hadm[0], 0))
    return (pos, neg)


if __name__ == '__main__':
    dx = pd.read_csv(filename_dx)
    admits = pd.read_csv(filename_admits)
    pts = pd.read_csv(filename_pts)
    icustays = pd.read_csv(filename_icustays)