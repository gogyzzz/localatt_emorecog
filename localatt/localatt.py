#xxx prognet.py

import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

def init_linear(m):
    if type(m) == nn.Linear:
        tc.nn.init.xavier_normal(m.weight)
        nn.init.constant(m.bias, 0)


# model = localatt(nin, nhid, ncell, nout)
class localatt(nn.Module):
    def __init__(self, featdim, nhid, ncell, nout):
        super(localatt, self).__init__()

        self.featdim = featdim
        self.nhid = nhid
        self.fc1 = nn.Linear(featdim, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        self.do2 = nn.Dropout()


        self.blstm = tc.nn.LSTM(nhid, ncell, 1, 
                batch_first=True,
                dropout=0.5,
                bias=True,
                bidirectional=True)

        self.u = nn.Parameter(tc.zeros((ncell*2,)))
        # self.u = Variable(tc.zeros((ncell*2,)))

        self.fc3 = nn.Linear(ncell*2, nout)

        self.apply(init_linear)

    def forward(self, inputs_lens_tuple):

        inputs = Variable(inputs_lens_tuple[0])
        batch_size = inputs.size()[0]
        lens = list(inputs_lens_tuple[1])

        indep_feats = inputs.view(-1, self.featdim) # reshape(batch) 

        indep_feats = F.relu(self.fc1(indep_feats))

        indep_feats = F.relu(self.do2(self.fc2(indep_feats)))

        batched_feats = indep_feats.view(batch_size, -1, self.nhid)

        packed = pack_padded_sequence(batched_feats, lens, batch_first=True) 

        output, hn = self.blstm(packed)

        padded, lens = pad_packed_sequence(output, batch_first=True, padding_value=0.0)

        alpha = F.softmax(tc.matmul(padded, self.u))

        return F.softmax((self.fc3(tc.sum(tc.matmul(alpha, padded), dim=1))))
