#!/usr/bin/env python

import os
import sys
import json as js

dataset = sys.argv[1]
exp = sys.argv[2]

# manual param.

lr=0.00005
ephs=200 # epochs
bsz=64 # batch size
measure="war"
featdim=32
nhid=512
ncell=128

# 

os.system('mkdir -p '+exp)

param = { 'dataset':dataset,
          'fulldata':'iemocap/utt_lld.pk.zero_mean.pk',
            'lr':lr,
            'ephs':ephs,
            'bsz':bsz,
            'measure':measure,
            'log':exp+'/log',
            'featdim':featdim,
            'nhid':nhid,
            'ncell':ncell,
            'model':exp+'/model.pth' }


with open(exp+'/param.json','w') as f:
    js.dump(param,f, sort_keys=True, indent=4)
    print(js.dumps(param, sort_keys=True, indent=4))

print('')
print('<',exp,'>')
os.system('ls '+exp)
