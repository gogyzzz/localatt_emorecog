#!/usr/bin/env python

import sys
import os
import numpy as np
import pickle as pk

from tqdm import tqdm

from fileutils.htk import readHtk

utt = sys.argv[1]
htklistpath = sys.argv[2]
outpk = sys.argv[3]

with open(utt) as f:
    uttlist = [e.strip() for e in f.readlines()]

with open(htklistpath) as f:
    htklist = [e.strip() for e in f.readlines()]

n_samples = len(htklist)
print('n_samples:',n_samples)

utt_lld = {}

for i in tqdm(range(n_samples)):
#for i in tqdm(range(10)):

    utt_lld[uttlist[i]]=readHtk(htklist[i])

print('')
print('<',outpk,'>')
print('utt_lld[uttlist[0]]')
print(np.shape(utt_lld[uttlist[0]]))
print(utt_lld[uttlist[0]])

pk.dump(utt_lld, open(outpk,'wb'))


