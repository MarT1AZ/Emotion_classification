from matplotlib import pyplot as plt
import os
import gc
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms.v2 as v2

from torch.utils.data import Dataset, Subset, DataLoader, random_split

import random
import pandas as pd

from model import FacialNet, FacialNet_finegrained
from dl_utils import train_one,test,multiclass_f1_score_aggregation

torch.manual_seed(1986)
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object
print(f'using DEVICE = {DEVICE}')


train_dir = 'train'
test_dir = 'test'
# v2.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

transforms_train = v2.Compose([
    v2.Resize((48,48)),
    v2.RandomHorizontalFlip(), # data augmentation
    v2.RandomAutocontrast(p = 1.0),
    v2.ToTensor() # normalization
])

transforms_val = v2.Compose([
    v2.Resize((48,48)),
    v2.ToTensor()
])


train_ds = datasets.ImageFolder(train_dir, transforms_train)
valid_ds = datasets.ImageFolder(test_dir, transforms_val) # use test folder for validation

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

class_names = train_ds.classes
# print('Train dataset size:', len(train_ds))
# print('Validation dataset size:', len(valid_ds))

# print('Class names:', class_names)

BALANCE = True
if(BALANCE):

  class_indice = {'angry':[], 'fear':[], 'happy':[], 'neutral':[], 'sad':[], 'surprise':[]}
  for idx in range(0,len(train_ds)):
    class_indice[class_names[train_ds[idx][1]]].append(idx)
  sampled_indices = []
  for class_ in class_names:
    sampled_indices = sampled_indices + random.sample(class_indice[class_], k=3100)
  train_ds = Subset(train_ds,sampled_indices)

  class_count = {'angry':0, 'fear':0, 'happy':0, 'neutral':0, 'sad':0, 'surprise':0}
  for idx in range(0,len(train_ds)):
    count = class_count[class_names[train_ds[idx][1]]]
    class_count[class_names[train_ds[idx][1]]] = count + 1


print('Train dataset size:', len(train_ds))
print('Validation dataset size:', len(valid_ds))

print('Class names:', class_names)


model = FacialNet_finegrained()
model.to(DEVICE)
loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
train_perf = {'loss':[],'accuracy':[],'f1':[]}
valid_perf = {'loss':[],'accuracy':[],'f1':[]}
save_at = []


best_vloss = 100000

EPOCHS = 35
for epoch in range(0,EPOCHS):
  train_one(DEVICE,train_dl,model,loss_fn,optimizer,epoch,loss_display_interval = 200)

  train_loss,res_b,res_c = test(DEVICE,train_dl,model,loss_fn,len(class_names))
  train_acc = res_b/len(train_ds)
  train_f1 = multiclass_f1_score_aggregation(res_c,len(class_names))

  valid_loss,res_b,res_c = test(DEVICE,valid_dl,model,loss_fn,len(class_names))
  valid_acc = res_b/len(valid_ds)
  valid_f1 = multiclass_f1_score_aggregation(res_c,len(class_names))
  print(f'train loss {train_loss}, train accuracy {train_acc:.5f}, train f1 score {train_f1:.5f}')
  print(f'valid loss {valid_loss}, train accuracy {valid_acc:.5f} ,train f1 score {valid_f1:.5f}')
  print('========================================================')

  if valid_loss < best_vloss:
        best_vloss = valid_loss
        torch.save(model.state_dict(), 'model_best_vloss.pth')
        save_at.append(epoch)
        print('Saved best model to model_best_vloss.pth')

  train_perf['loss'].append(train_loss)
  train_perf['accuracy'].append(train_acc)
  train_perf['f1'].append(train_f1)
  valid_perf['loss'].append(valid_loss)
  valid_perf['accuracy'].append( valid_acc)
  valid_perf['f1'].append(valid_f1)


train_perf_df = pd.DataFrame(train_perf)
valid_perf_df = pd.DataFrame(valid_perf)
train_perf_df.to_csv('train_perf.csv')
valid_perf_df.to_csv('valid_perf.csv')


print('\nsample testing')
model = FacialNet_finegrained()
model.load_state_dict(torch.load('model_best_vloss.pth'))
model.eval()


sample_image = valid_ds[3][0]
sample_image = torch.unsqueeze(sample_image,dim =0)


sample = np.random.choice(len(valid_ds),8)
sample_test_images = []
sample_test_labels = []
for idx in sample:
    sample_test_images.append(valid_ds[idx][0])
    sample_test_labels.append(valid_ds[idx][1])
    test_batch = torch.stack(sample_test_images)
with torch.no_grad():
    model.eval()
    probs = nn.functional.softmax(model(test_batch))
    prediction = probs.argmax(dim = 1)

for sidx in  range(0,len(sample_test_labels)):
    print(f'predicted : { class_names[prediction[sidx]] } actual : {class_names[sample_test_labels[sidx]]}')



