# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:02:41 2017

@author: af1tang
"""
import pandas as pd
import numpy as np

filename = '/media/sf_mimic/csv/DIAGNOSES_ICD.csv/DIAGNOSES_ICD_DATA_TABLE.csv'

def group_icd9 (dx):
    subj  = list(set(dx.SUBJECT_ID))
    #make group ICD9 codes with first 3 chars only
    icd = list(set(list(dx.ICD9_CODE)))
    icd = [x for x in icd if str(x) != 'nan']
    groups = []
    for i in icd:
        groups.append(i[0:3])
    groups = sorted(list(set(groups)))

    #make default feature vector and index dictionary
    index = {k:v for v,k in enumerate (groups)}
    #make concatenated feature vectors for each patient
    for s in subj:
        feature = [0]*len(groups)
        lst = list(dx[dx['SUBJECT_ID'] == s].ICD9_CODE)
        lst = [x for x in lst if str(x) !='nan']
        lst = [x[0:3] for x in lst]
        for i in lst:
            feature[index[lst[0]]] += 1
        dct[s] = feature

    return(dct, subj, index, groups)
    
if __name__ == '__main__':
    dx = pd.read_csv(filename)