#xxx train.py

import os
from toolz import curry
from tqdm import tqdm
from torch.autograd import Variable


# on pytorch

@curry
def validate_loop_lazy(name, __validate, loader, log):

    losses = [0.0] * len(loader)
    scores = [0.0] * len(loader)

    for i, batch in enumerate(tqdm(loader, total=len(loader))):

        losses[i], scores[i]= __validate(batch)

    if len(loader) > 1:
        score = sum(scores[:-1])/(len(scores[:-1]))
        loss = sum(losses[:-1])/(len(losses[:-1]))

    else:
        score = scores[0]
        loss = losses[0]

    command = '[%s] score: %.3f, loss: %.3f'%(name, score, loss)
    os.system('echo "%s" >> %s'%(command, log))
        
    return loss, score


def train(model, loader, _valid_lazy, valid_loop, crit, optim, ephs, log):

    best_valid_score = 0.0
    best_model = model

    for epoch in tqdm(range(ephs), total=ephs):
        for i, batch in enumerate(loader):


            inputs = batch[0]
            targets = batch[1]
            
            optim.zero_grad()
            model.train() # autograd on

            train_loss = crit(model(inputs), targets)
            train_loss.backward()

            optim.step()
            model.eval() # autograd off

            __val_lz = _valid_lazy(model=model)

        command ='[train] %4d/%4dth epoch, loss: %.3f'%(
                epoch, ephs, train_loss.data[0])
        os.system('echo "%s" >> %s'%(command, log))

        valid_loss, valid_score = valid_loop(__validate=__val_lz)

        if valid_score > best_valid_score:

            best_valid_score = valid_score

            command = '[valid] bestscore: %.3f, loss: %.3f'%(
                    valid_score, valid_loss)
            os.system('echo "%s" >> %s'%(command, log))
            best_model = model

    print('Finished Training')

    return best_model

