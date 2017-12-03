# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:00:14 2017

@author: andy
"""

# Server:Folder:: tangfeng@illidan-gpu-1:/mnt/research/data/OHSU-IRB-17110

import pandas as pd
import numpy as np
import pickle
import random
import gensim
import matplotlib.pyplot as plt
from itertools import tee

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC as SVM
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, f1_score

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Activation, Dropout, TimeDistributed, Bidirectional, Masking, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras.preprocessing import sequence
from keras import metrics

f_amb = 'Dodge_eirb17110_Ambulatory_Encounters.csv' #11G
f_dxlst = 'Dodge_eirb17110_Diagnoses_Problem_List.csv' #6G
f_mar = 'Dodge_eirb17110_Medications_MAR.zip'
f_labs = 'Dodge_eirb17110_Lab_Results.csv'#8g
f_micro = 'Dodge_eirb17110_Microbiology_Results.csv' #300mb
f_hosp = 'Dodge_eirb17110_Hospital_Encounters.csv' #600mb
f_dx = 'Dodge_eirb17110_Diagnoses_Encounters.csv' #6G
f_hx = 'Dodge_eirb17110_demogr.csv' #100mb
f_orders = 'Dodge_eirb17110_Orders.csv' #4g
f_icd = '/sample/eirb17110_diagnoses_part1.xlsx'
f_demo = '/sample/eirb17110_demographics.xlsx'
#31M	./sample
f_med = 'Dodge_eirb17110_Medications_Current_List.csv' #31gb
f_medor= 'Dodge_eirb17110_Medications_Ordersr.csv'  #4.4G
f_vitals = 'Dodge_eirb17110_Vital_Signs.csv' #1gb
f_surg = 'Dodge_eirb17110_Surgeries.csv' #22mb
root = '/mnt/research/data/OHSU-IRB-17110/'

def numbers(filename):
    chunksize = 10**6
    #pts = {}; los = []; dx = {} #admission and dx data
    #eth = []; ages = []; insur = []; gender ={'MALE':0, 'FEMALE':0, 'UNKNOWN': 0, 'RDW UNK':0} #demo
    aadm = {}; types ={}; insur = {}; ages = {}
    #mort_h = []; mort_30 = []; readm = []
    
    with open(filename, 'r') as f: 
        for numlines, _ in enumerate(f): pass
    numlines +=1
    #53687572 for f_amb, 32403450 for f_dx, 35459836 for f_dxlst
    #12240680 for f_vitals, 84340688 for f_labs
    cols = list(pd.read_csv(filename, nrows=0).columns)
    for i in range(0, numlines, chunksize):
        print(i)
        df = pd.read_csv(filename, header = None, nrows = chunksize, skiprows = i, names = cols)        
        #dx = extract_values(df, dx)
        #del df
        #ambulatory
        enc_id= dict(df.groupby('HASH_SUBJECT_ID')['HASH_ENC_ID'].nunique())
        enc_type = dict(df.groupby('ENC_TYPE')['HASH_SUBJECT_ID'].nunique())
        grp = dict(df.groupby('INSURANCE_GROUP')['HASH_SUBJECT_ID'].nunique())
        yrs = df[~df['HASH_SUBJECT_ID'].isin(list(ages.keys()))].groupby('HASH_SUBJECT_ID')['AGE_YRS'].apply(lambda df: df.sample(1))
        for (k,v) in enc_id.items():
            if k in aadm.keys():
                aadm[k] += v
            else:
                aadm[k] = v
        for (k,v) in enc_type.items():
            if k in types.keys():
                types[k] += v
            else:
                types[k] = v
        for (k,v) in grp.items():
            if k in insur.keys():
                insur[k] += v
            else:
                insur[k] = v
        d = {k:yrs.ix[k].values[0] for k in yrs.index.levels[0]}
        ages.update(d)
        del df, enc_id, enc_type, grp, yrs
    return (aadm, types, insur, ages)
        

def neuro_subj(dx, neuro):
    '''@pre: dict dx has diagnostic history of patients in format {s: [d1, d2 ...]}
    neuro is a list containing neurodegen icd9's
    @post: generates list of subjects containing neurodegen icd9s'''
    subj = []
    for (k,v) in list(dx.items()):
        if len(set(v).intersection(set(neuro))) > 0:
            subj.append(k)
    return (subj)


def hosp_enc(filename):
    df = pd.read_csv(filename)
    #hospital 840660 total admits
    hadm = df.groupby('HASH_SUBJECT_ID')['HASH_ENC_ID'].unique()
    los = list(df.INPATIENT_LENGTH_OF_STAY)
    return (hadm, los)

def get_labs(df, subj):
    pts = list(set(df.HASH_SUBJECT_ID))
    intersect = list(set(pts).intersection(set(subj)))
    return (intersect)

def extract_values (df, item_lst, label = 'COMPONENT_NAME', val_label = 'RESULT_TEXT'):
    '''
    @pre: slice or entire dataframe df, with desired feature to be extracted (label) and corresponding value column (val_label)
    @post: hash table with key = label and value = val_label list
    redundancies are eliminated; hash table contains only unique values of val_label.
    '''
    tmp = df.groupby(label)[val_label].unique().apply(list).to_dict()
    for item in tmp.items():
        if item[0] in item_lst.keys():
            item_lst[item[0]] += item[1]
            item_lst[item[0]] = list(set(item_lst[item[0]]))
        else:
            item_lst[item[0]] = item[1]
    return (item_lst)
    
def value_counts(df, counts, label='COMPONENT_NAME', val_label = 'RESULT_TEXT'):
    tmp = df.groupby(label)[val_label].count()
    for item in tmp.items():
        if item[0] in counts.keys():
            counts[item[0]] += item[1]
        else:
            counts[item[0]] = item[1]
    return(counts)

def get_vitals(df, subj):
    '''format of pt vector: {s: {h: [t, bp1, bp2, hr, tmp, rr, spo2], ...}, ...}
    '''
    pts = list(set(df.HASH_SUBJECT_ID))
    intersect = list(set(pts).intersection(set(subj)))
    return (intersect)

def generate_neuro(df, neuro, subj):
    '''
    @pre: subject list of neurodegen pts
    @post: returns dataframe of vital signs of neurodegen pts only
    '''
    pts = list(set(df.HASH_SUBJECT_ID))
    intersect = list(set(pts).intersection(set(subj)))
    tab = df[df['HASH_SUBJECT_ID'].isin(intersect)]
    if len(neuro) <1:
        neuro = tab
    else: 
        neuro = pd.concat([neuro, tab])
    return (neuro)

def histplot(x, title = "", xlabel = "", ylabel = ""):
    hist, bins = np.histogram(x, bins=50)
    y = [10*i/sum(hist) for i in hist]
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, y, align='center', width=width)
    plt.xlabel(xlabel, fontsize = 20)
    plt.ylabel(ylabel, fontsize = 20)
    plt.title(title, fontsize = 24)
    plt.show()

def pie(dct):
    fig, ax = plt.subplots()
    x = np.array(list(dct.keys()))
    y = np.array(list(dct.values()))
    colors = ['yellowgreen','red','gold','lightskyblue','white','lightcoral','blue','pink', 'darkgreen','yellow','grey','violet','magenta','cyan']
    porcent = 100.*y/y.sum()
    
    patches, texts = plt.pie(y, colors=colors, startangle=90, radius=1.2)
    labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, porcent)]
    
    sort_legend = True
    if sort_legend:
        patches, labels, dummy =  zip(*sorted(zip(patches, labels, y),
                                              key=lambda x: x[2],
                                              reverse=True))
    
    plt.legend(patches, labels, loc='right center', bbox_to_anchor=(-0.1, 1.),
               fontsize=8)
    plt.show()
    #plt.savefig('piechart.png', bbox_inches='tight')


##### Label Generation ######
def generate_mci(dx):
    lst = sorted(dx.items(), key = lambda x: x[1], reverse = True)
    mci = [i for i in lst if '331.83' in i[0]]
    mci += [i for i in lst if '331.9' in i[0]]
    mci += [i for i in lst if '331.89' in i[0]]
    return (mci)

def generate_MCI_AD_Y(df, mci, ad, pts, icd9 = 'ICD9_CODE', date = 'DX_DATE'):
    MCI = df[df[icd9].isin(mci)].groupby('HASH_SUBJECT_ID')[date].first().to_dict()
    AD = df[df[icd9].isin(ad)].groupby('HASH_SUBJECT_ID')[date].first().to_dict()
    for (k,v) in MCI.items():
        if k in pts.keys():
            if (pts[k]['MCI'] == None) or (v < pts[k]['MCI']):
                pts[k]['MCI'] = v
        else:
            pts[k] = {'MCI':v, 'AD': None}
            
    for (k,v) in AD.items():
        if k in pts.keys():
            if (pts[k]['AD'] == None) or (v < pts[k]['AD']):
                pts[k]['AD'] = v
        else:
            pts[k] = {'MCI': None, 'AD':v}    
    return(pts)

def dx_histories(df, X, groups, icd9 = 'ICD9_CODE', date = 'DX_DATE'):
    #groups = dict((v, k) for (k,v) in groups.items())
    subj = list(set(df.HASH_SUBJECT_ID))
    for s in subj:
        tmp = df[df['HASH_SUBJECT_ID'] == s][['HASH_ENC_ID', date, icd9]].to_dict(orient = 'split')['data']
        #0 index is Encounter ID
        #1 is DX_Date
        #2 is ICD9 Code
        hadm = list(set(map(lambda x:x[0], tmp)))
        piv_tab = [[x]+ [groups[y[2][0:3]] for y in tmp if (y[0]==x and y[2][0:3] in groups.keys())]+list(set(map(lambda y: y[1], filter(lambda z: z[0] == x, tmp)))) for x in hadm]
        #for i in range(len(piv_tab)):
        #    for j in range(len(piv_tab[i][1:-1])):
        #        try:
        #            piv_tab[i][j+1] = groups[piv_tab[i][j+1][0:3]]
        #        except:
        #            print(piv_tab[i][j+1])
        if s in X.keys():
            X[s] += piv_tab
        else:
            X[s] = piv_tab
    return (X)

#### Feature Generation #####
def X_flat(DX, subj, window = 1.0, size = 1035):
    '''
    @pre: subj is hash table in {s: {y: 0/1, t: time} ... } format for each subject s in subset S. 
    - DX is hash table with hadm's for each s: {s: [h1, [idx, idx...], t1], [h2...]}.
    - window is float prediction window:
        {6m: 0.5, 12m: 1.0, 24m: 2.0, 36m: 3.0}
    - size is int size of one-hot vectors to be converted per index.
    @post: returns hash table X w/ summary dx vector for each s: {s: [0,2,1, ...] ...}
    '''
    X = X_tensor(DX, subj, window, size)
    X = {key: [sum(i) for i in zip(*X[key])] for key in list(X.keys())}
    return (X)

def X_tensor(DX, subj, window = 1.0, size = 1035):
    '''
    @pre: subj is hash table in {s: {y: 0/1, t: time} ... } format for each subject s in subset S. 
    - DX is hash table with hadm's for each s: {s: [h1, [idx, idx...], t1], [h2...]}.
    - window is float prediction window:
        {6m: 0.5, 12m: 1.0, 24m: 2.0, 36m: 3.0}
    - size is int size of one-hot vectors to be converted per index.
    @post: returns hash table X w/ matrices of summary dx vectors per hadm for each s: {s: [0,0,1,...], [0,2,0,...] ...}
    '''
    wanted = list(set(DX.keys()).intersection(set(subj.keys())))
    X = {key: [[sum(l) for l in zip(*[idx_2_OHV(i,size) for i in lst[1:-1]])] for lst in sorted(DX[key], key = lambda x: x[-1]) if lst[-1] <= (subj[key]['T'] - window)] for key in wanted}
    return X

def X_W2V(DX, subj, groups, window = 1.0, w2v_dim = 200, slider = 5):
    '''
    @pre: subj is hash table in {s: {y: 0/1, t: time} ... } format for each subject s in subset S. 
    - DX is hash table with hadm's for each s: {s: [h1, [idx, idx...], t1], [h2...]}.
    - groups is a reference hash table of indices of ICD9 group codes. 
    - window is float prediction window:
        {6m: 0.5, 12m: 1.0, 24m: 2.0, 36m: 3.0}
    size is int size of one-hot vectors to be converted per index.
    @post: returns hash table X w/ matrices of Word2Vec embedding of each dx vector per hadm for per s: {s: [w2v1], [w2v2] ...}
    - Uses skip-gram learning.
    - Also returns SG (model)
    - weights of hidden layer
    - vocab reference table of indices 
    '''
    wanted = list(set(DX.keys()).intersection(set(subj.keys())))
    groups = dict((v, k) for (k,v) in groups.items())
    flatten = lambda l: [item for sublist in l for item in sublist]
    DX_tr = {key: flatten([[groups[code] for code in visit[1:-1]] for visit in DX[key] if visit[-1] <= (subj[key]['T'] - window)]) for key in wanted}
    
    #train W2V model
    x_train = [i for i in list(DX_tr.values()) if i]
    SG = gensim.models.Word2Vec(sentences = x_train, sg = 1, size = w2v_dim, window = slider, min_count = 0, hs = 1, negative = 0)
    weights = SG.wv.syn0
    vocab = dict([(k, v.index) for k, v in SG.wv.vocab.items()])
    w2i, i2w = vocab_index(vocab)
    
    #Convert relevant DX
    for (k,v) in list(DX_tr.items()):
        v = list(map(lambda i: w2i[i] if i in w2i.keys() else 0, v))
        one_hot = np.zeros((len(v), weights.shape[0]))
        one_hot[np.arange(len(v)), v] = 1
        one_hot = np.sum(one_hot, axis= 0)
        DX_tr[k] = np.dot(one_hot.reshape(one_hot.shape[0]), weights)
    return (DX_tr, SG, weights, vocab)

##### KFOLD CV Pipeline #####
def kv_pipe(DX, labels, groups, tr_options, report_options):
    '''
    @pre: DX hash table database of dx histories for each patient per encounter. 
    DX codes are represented by index in ICD9 groups dictionary.
    - labels is hash table in {s: {y: 0/1, t: time} ... } format for each subject s in subset S. 
    - groups is a reference hash table of indices of ICD9 group codes. 
    - tr_options is a hash table with options for X_training options (e.g., flat, window size, ...)
    - report_options is a hash_table with options for result reporting 
    @post: pickle or h5 format for model and hash table of training and testing data
    '''
    random.seed(11)
    skf = StratifiedKFold(n_splits=5, random_state = 7)
    splits = [(i[0], list(i[1].items())[1][1]) for i in list(labels.items())]
    x_split, y_split = zip(*[(a, b) for a,b in splits])
    x_split, y_split = np.array(x_split), np.array(y_split)
    
    for train_index, test_index in skf.split(x_split, y_split):
        #1. downsample training split
        pos = [i for i in train_index if y_split[i] == 1]
        neg = random.sample([i for i in train_index if y_split[i] == 0], len(pos))
        tr_split = x_split[neg+pos]; random.shuffle(tr_split)
        te_split = x_split[test_index]
        
        #2. split labels hash table by kfold
        tr_subj = {key: labels[key] for key in tr_split}
        te_subj = {key: labels[key] for key in te_split}
        
        #3. generate train = [(x_tr,y_tr), ...] and test = [(x_te, y_te), ...]
        train, test = generate_x_y(DX, tr_subj, te_subj)
        x_tr, y_tr = zip(*[(a,b) for a,b in train])
        x_te, y_te = zip(*[(a,b) for a,b in test])
        
        #4. train models
        lr = LR(warm_start = True, C = 1e-3, penalty = 'l2', verbose = 1)    #sag if multiclass/multilabel
        svm = SVM(C=1e4, kernel = 'linear', verbose = True, probability = True, max_iter= 1000)
        rf = RF(warm_start = True, n_estimators = 450, verbose = 1)
        
    return

def generate_x_y(DX, tr_subj, te_subj):
    '''
    @pre: hash table X with subjects 
    '''
    X_tr = X_flat(DX, tr_subj)
    X_te = X_flat(DX, te_subj)
    train = [(v, tr_subj[k]['Y']) for k,v in list(X_tr.items())]
    test = [(v, tr_subj[k]['Y']) for k,v in list(X_te.items())]
    return (train, test)

##### Models #####


##### Utilities #####

def to_pickle(filename, dat):
    with open(filename, 'wb') as f:
        pickle.dump(dat, f)

def idx_2_OHV (idx, size):
    tmp = [0]*size
    tmp[idx] = 1
    return (tmp)

def vocab_index (vocab):
    word2idx = vocab
    idx2word = dict([(v,k) for k, v in vocab.items()])
    return (word2idx, idx2word)

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
        hadm = list(set(admits[admits['HASH_SUBJECT_ID']==s].HASH_ENC_ID.values))
        if len(hadm) <1: 
            pass
        else:   
            t = [(admits[admits['HASH_ENC_ID']==i]['HOSPITAL_ADMIT_TIME'].values[0], i) for i in hadm]
            #dt = [(pd.to_datetime(admits[admits['HADM_ID']==i]['DISCHTIME'].values[0]), i) for i in hadm]
            t = sorted(t, reverse = True)
            dt = {}
            for i in hadm:
                dt[i] = admits[admits['HASH_ENC_ID']==i]['HOSPITAL_DISCHARGE_TIME'].values[0]
            for t2,t1 in pairwise(iterable = t):
                try:
                    if (t2[0]-dt[t1[1]])*12<=30:
                        plus.append((s, t1[1], 1))
                    elif (t1[1] in HADM):
                        minus.append((s, t1[1], 0))
                except: pass
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
                if hadm[0] in HADM:
                    neg.append((s, hadm[0], 0))
    return (pos, neg)

if __name__ == '__main__':
    filename = root + f_amb
    aadm, types, insur, ages = numbers(filename)
    to_pickle('aadm.pkl', aadm)
    to_pickle('types.pkl', types)
    to_pickle('insur.pkl', insur)
    to_pickle('ages.pkl', ages)


'''
#SCRATCH WORK 

#Flatten Trick
flatten = lambda l: [item for sublist in l for item in sublist]

#dict with Uneven Arrays Trick
df_neuro = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in dx_neuro.items()]), columns = ['HASH_SUBJECT_ID','ICD_CODES'])

#group list by values
values = set(map(lambda x:x[1], list))
newlist = [[y[0] for y in list if y[1]==x] for x in values]


def count_dxs(df, dx):
    dct = df.groupby('HASH_SUBJECT_ID')['DX_ICD'].unique().apply(list).to_dict()
    for item in dct.items():
        if item[0] in dx.keys():
            dx[item[0]]+= item[1]
            dx[item[0]] = list(set(dx[item[0]]))
        else:
            dx[item[0]] = item[1]
    return (dx)

def count_hx(df, hx):
    dct = df.groupby('HASH_SUBJECT_ID')['HASH_ENC_ID'].unique().apply(list).to_dict()
    for item in dct.items():
        if item[0] in hx.keys():
            hx[item[0]]+= item[1]
            hx[item[0]] = list(set(hx[item[0]]))
        else:
            hx[item[0]] = item[1]
    return (hx)
    '''
