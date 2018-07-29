
from sklearn.utils.class_weight import compute_class_weight
import torch as tc
import pickle as pk
import numpy as np

def get_class_weight(dataset_utt_lab_path, device):
    with open(dataset_utt_lab_path) as f:
        y = [int(l.split()[1]) for l in f.readlines()]

    cls = list(set(y))
    y = np.int_(np.array(y))

    cls_wgt = compute_class_weight('balanced', cls, y)
    return tc.FloatTensor(cls_wgt).to(device)
