#xxx forward_lazy.py

from toolz import curry
from sklearn.metrics import recall_score

@curry
def validate_war_lazy(batch, model, crit):
    inputs = batch[0] # expected padded 
    targets = batch[1]
    sample_wgts = batch[2]

    outputs = model(inputs)
    loss = crit(outputs, targets).data
    score = recall_score(
        outputs.max(dim=1)[1].data.cpu().numpy(),
        targets.data.cpu().numpy(),
        average='weighted',
        sample_weight=sample_wgts)

    return loss, score

@curry
def validate_uar_lazy(batch, model, crit):
    inputs = batch[0] # expected padded 
    targets = batch[1]
    sample_wgts = batch[2]

    outputs = model(inputs, lens)
    loss = crit(outputs, targets)

    score = recall_score(
        outputs.max(dim=1)[1].data.cpu().numpy(),
        targets.data.cpu().numpy(),
        average='macro')

    return loss, score

