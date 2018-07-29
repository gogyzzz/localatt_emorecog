#!/usr/bin/env python 
import os
import json as js
import argparse as argp
import pickle as pk

import torch as tc
import torch.nn as nn
from torch.utils.data import DataLoader

from localatt.get_class_weight import get_class_weight
from localatt.lld_dataset import lld_dataset
from localatt.lld_dataset import lld_collate_fn
from localatt.validate_lazy import validate_war_lazy 
from localatt.validate_lazy import validate_uar_lazy
from localatt.localatt import localatt
from localatt.train import train
from localatt.train import validate_loop_lazy

device=tc.device("cuda:0" if tc.cuda.is_available() else "cpu")

pars = argp.ArgumentParser()
pars.add_argument('--propjs', help='property json')
with open(pars.parse_args().propjs) as f:
    #p = js.load(f.read())
    p = js.load(f)

print(js.dumps(p, indent=4))

with open(p['fulldata'], 'rb') as f:
    full_utt_lld = pk.load(f)

train_utt_lab_path    = p['dataset']+'/train_utt_lab.list'
dev_utt_lab_path      = p['dataset']+'/dev_utt_lab.list'
eval_utt_lab_path      = p['dataset']+'/eval_utt_lab.list'

model_pth    =  p['model']
log          =  p['log']

lr=p['lr']
ephs=p['ephs']
bsz=p['bsz']
    
with open(p['dataset']+'/idx_label.json') as f:
    tgt_cls = js.load(f) # target classes

featdim=p['featdim']
nhid=p['nhid']
measure=p['measure']
ncell=p['ncell']

nout = len(tgt_cls)

cls_wgt = get_class_weight(train_utt_lab_path,device) # done.
print('class weight:', cls_wgt)

valid_lazy = {'uar': validate_uar_lazy,
                'war': validate_war_lazy } # done.

# loading

trainset = lld_dataset(full_utt_lld, train_utt_lab_path, cls_wgt, device)
devset = lld_dataset(full_utt_lld, dev_utt_lab_path, cls_wgt, device)
evalset = lld_dataset(full_utt_lld, eval_utt_lab_path, cls_wgt, device)

_collate_fn = lld_collate_fn(device=device) # done.
trainloader = DataLoader(trainset, bsz, collate_fn=_collate_fn)
devloader = DataLoader(devset, bsz, collate_fn=_collate_fn)
evalloader = DataLoader(evalset, bsz, collate_fn=_collate_fn)

# training
model = localatt(featdim, nhid, ncell, nout) # done.
model.to(device)
print(model)
crit = nn.CrossEntropyLoss(weight=cls_wgt)

if os.path.exists(model_pth):

    print(model_pth, 'already exists')

else:

    optim = tc.optim.Adam(model.parameters(), lr=0.00005)

    _val_lz = valid_lazy[measure](crit=crit)
    _val_loop_lz = validate_loop_lazy(name='valid', 
            loader=devloader,log=log)

    trained = train(model, trainloader, 
            _val_lz, _val_loop_lz, crit, optim, ephs, log)

    tc.save(trained.state_dict(), model_pth)

model.load_state_dict(tc.load(model_pth))
_val_lz = valid_lazy[measure](model=model, crit=crit)

# testing
validate_loop_lazy('test', _val_lz, evalloader, log) 
