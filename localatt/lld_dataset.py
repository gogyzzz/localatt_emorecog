#xxx egemaps_dataset.py

import pickle as pk
import numpy as np
import torch as tc
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from toolz import curry

class lld_dataset(Dataset):

    def __init__(self, utt_lld, utt_lab_path, cls_wgt, device):

        with open(utt_lab_path) as f:
            utt_lab = [l.strip().split() for l in f.readlines()]
            # utt label
        
        self.samples = [0] * len(utt_lab)

        for i, pair in enumerate(utt_lab):
            utt, label = pair
            label = int(label)

            lld = utt_lld[utt]

            self.samples[i] = (
                Variable(tc.FloatTensor(lld)).to(device),# input
                len(lld),# len
                Variable(tc.LongTensor(np.array([label]))).to(device), # label
                cls_wgt[label]) # class weight

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

@curry
def lld_collate_fn(data, device):

    data.sort(key=lambda x: x[1], reverse=True)
    llds, lens, labels, wgts = zip(*data)

    padded_llds = tc.zeros([len(llds), max(lens), llds[0].size()[1]]).to(device)
    for i, lld in enumerate(llds):
        padded_llds[i, 0:lens[i], :] = lld
        
    return ((padded_llds, lens), 
            Variable(tc.stack(labels), requires_grad=False).view(-1), 
            wgts)
