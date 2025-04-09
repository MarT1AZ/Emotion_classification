import torch
import numpy as np
from torcheval.metrics.functional import multiclass_accuracy




def multiclass_confusion_value(predict, true, num_class):
  # predict : shape = (batch)
  # true : shape = (batch)
  # num_class : int
  class_confusion_dict = {}

  for cidx in range(0,num_class):
    class_confusion_dict[cidx] = {'TP':0,'TN':0,'FP':0,'FN':0}

  for cidx in range(0,num_class):
    for idx in range(0,predict.shape[0]):
      if(predict[idx] == true[idx] and predict[idx] == cidx):
        class_confusion_dict[cidx]['TP'] = class_confusion_dict[cidx]['TP'] + 1
      elif(predict[idx] != true[idx] and predict[idx] == cidx):
        class_confusion_dict[cidx]['FP'] = class_confusion_dict[cidx]['FP'] + 1
      elif(predict[idx] == true[idx] and predict[idx] != cidx):
        class_confusion_dict[cidx]['FN'] = class_confusion_dict[cidx]['FN'] + 1
      else:
        class_confusion_dict[cidx]['TN'] = class_confusion_dict[cidx]['TN'] + 1
  return class_confusion_dict


def multiclass_f1_score_aggregation(confusion_values_dict,num_class):

  f1_list = []
  for cidx in range(0,num_class):
    TP = confusion_values_dict[cidx]['TP']
    FP = confusion_values_dict[cidx]['FP']
    FN = confusion_values_dict[cidx]['FN']
    f1_list.append(2 * TP/ (2 * TP + FP + FN))

    # return the micro average
  return np.array(f1_list).mean()




def train_one(device,dl,model,loss_fn,optim, epoch,loss_display_interval):
  model.train()
  total_loss = 0.0
  run_loss = 0.0
  num_batch = len(dl)
  for i, batch in enumerate(dl):
    # X is tensor in shape (batch, channel = 3, width, height)
    # y is a label tensor in shape (batch, 1)
    # loss function is cross-entrophy

    X, y = batch
    X, y = X.to(device), y.to(device)

    optim.zero_grad()
    pred = model(X)
    # pred is in shape (batch, len_class) aka one hot encoded

    loss = loss_fn(pred, y)
    loss.backward()
    optim.step()

    run_loss = run_loss + loss.item()/len(batch)

    if (i + 1) % loss_display_interval == 0:
      total_loss = total_loss + run_loss/loss_display_interval
      print(f'epoch = {epoch} : progress {i} / {num_batch} ')
  print(f'final loss for epoch {epoch} :{run_loss / 1000}')

def test(device,dl,model,loss_fn,num_class):
  num_class
  model.eval()
  total_loss = 0.0
  total_accurate = 0.0
  total_sample = 0

  class_confusion_dict = {}

  for cidx in range(0,num_class):
    class_confusion_dict[cidx] = {'TP':0,'TN':0,'FP':0,'FN':0}

  with torch.no_grad():
    for i, batch in enumerate(dl):
      # X is tensor in shape (batch, channel = 3, width, height)
      # y is a label tensor in shape (batch, 1)
      # loss function is cross-entrophy

      X, y = batch
      X, y = X.to(device), y.to(device)
      pred = model(X)
      # pred is in shape (batch, len_class) aka one hot encoded

      loss = loss_fn(pred, y).item()
      pred = pred.argmax(1)
      total_accurate = total_accurate + (pred == y).sum().item()

      total_loss = total_loss + loss
      total_sample = total_sample + len(y)
      # accurate_count = multiclass_accuracy(pred,
      #                               y,
      #                               average = 'micro') * dl.batch_size
                                    # should be integer between 0 to 64 no less no more
                                    # use micro averaging to get the global number of prediction right reggardless of class

  

      running_confusion_values = multiclass_confusion_value(pred, y, num_class)

      for cidx in  range(0,num_class):
        for s in ['TP','TN','FP','FN']:
          class_confusion_dict[cidx][s] = class_confusion_dict[cidx][s] + running_confusion_values[cidx][s]

  return total_loss/len(dl), total_accurate/total_sample, class_confusion_dict

