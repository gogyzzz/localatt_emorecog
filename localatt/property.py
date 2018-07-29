# property.py

device=tc.device("cuda:0" if tc.cuda.is_available() else "cpu")

pretrain_pk = 'iemocap/tmp2/pretrain.pk'
train_pk    = 'iemocap/tmp2/train.pk'
predev_pk   = 'iemocap/tmp2/predev.pk'
dev_pk      = 'iemocap/tmp2/dev.pk'
eval_pk     = 'iemocap/tmp2/eval.pk'

dnn_pth = 'iemocap/tmp2/exp/premodel.pth'
prognet_pth    = 'iemocap/tmp2/exp/model.pth'

lr=0.00005
preephs=150
ephs=300
bsz=512

dnn_cls={0:'Male',1:'Female'}
prognet_cls={0:'Happiness', 1:'Sadness', 2:'Neutral', 3:'Anger'}

nin=88
nhid=256
dnn_nout=len(dnn_cls)
prognet_nout=len(prognet_cls)

measure='war'
