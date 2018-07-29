#!/usr/bin/env python

import os
import sys
import pandas as pd
import json as js
import numpy as np
import pickle as pk

csv = sys.argv[1] 
utt_lld_pk = sys.argv[2]
dataset = sys.argv[3]

# manual param.

devfrac=0.2
session=1
featdim=32

# make 

full = pd.read_csv(csv)
print(full.head())
#print(full.columns)
#print(set(full.loc[:,'gender']))
label_set = set(full.loc[:,'emotion'])

idx_label_dict = {i:label for i, label in enumerate(label_set)}

os.system('mkdir -p '+dataset)

with open(dataset+'/idx_label.json','w') as f:
    print(js.dumps(idx_label_dict, sort_keys=True, indent=4),file=f)

os.system('head %s/*.json'%(dataset))

# make dataset dataframe

label_idx_dict = {label:i for i, label in enumerate(label_set)}

_eval = full.loc[full.loc[:,'session'].isin([session])]
_traindev = full.loc[~full.loc[:,'session'].isin([session])]

_dev = pd.concat([_traindev.loc[_traindev.loc[:,'emotion'] == 'N'].sample(frac=devfrac),
    _traindev.loc[_traindev.loc[:,'emotion'] == 'A'].sample(frac=devfrac),
    _traindev.loc[_traindev.loc[:,'emotion'] == 'S'].sample(frac=devfrac),
    _traindev.loc[_traindev.loc[:,'emotion'] == 'H'].sample(frac=devfrac)],
    ignore_index=True)

_train = _traindev.loc[~_traindev.loc[:,'utterance'].isin(_dev.loc[:,'utterance'])]

print('')
print('number of samples fullset:',len(full))
print('number of samples traindev:',len(_traindev))
print('number of samples train:',len(_train))
print('number of samples dev:',len(_dev))
print('number of samples eval:',len(_eval))
print('')

# make dataset pk

utt_lld = pk.load(open(utt_lld_pk,'rb'))
train_mats = [0] * len(_train)

train_labels = [0] * len(_train)
dev_labels = [0] * len(_dev)
eval_labels = [0] * len(_eval)


for i, irow in enumerate(_train.iterrows()):
    _, row = irow
    train_labels[i] = label_idx_dict[row['emotion']]
 
    train_mats[i] = utt_lld[row['utterance']]

for i, irow in enumerate(_dev.iterrows()):
    _, row = irow
    dev_labels[i] = label_idx_dict[row['emotion']]

for i, irow in enumerate(_eval.iterrows()):
    _, row = irow
    eval_labels[i] = label_idx_dict[row['emotion']]

# zero mean 

train_utt_means = np.ndarray(shape=(len(train_mats),featdim))

for i, mat in enumerate(train_mats):
    train_utt_means[i] = np.mean(mat, axis=0)

train_mean = np.mean(train_utt_means,axis=0)

print('train_mean shape:',np.shape(train_mean))

for utt, mat in utt_lld.items():
    utt_lld[utt] = mat - train_mean

with open(utt_lld_pk +'.zero_mean.pk', 'wb') as f:
    pk.dump(utt_lld, f)

with open(dataset+'/train_utt_lab.list','w') as f:
    for utt, lab in zip(list(_train.loc[:,'utterance']), train_labels):
        print('%s %d'%(utt,lab),file=f)

with open(dataset+'/dev_utt_lab.list','w') as f:
    for utt, lab in zip(list(_dev.loc[:,'utterance']), dev_labels):
        print('%s %d'%(utt,lab),file=f)

with open(dataset+'/eval_utt_lab.list','w') as f:
    for utt, lab in zip(list(_eval.loc[:,'utterance']), eval_labels):
        print('%s %d'%(utt,lab),file=f)

print('')
print('<',dataset,'>')
os.system('ls '+dataset)

