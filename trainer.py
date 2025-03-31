from torch.utils.tensorboard import SummaryWriter
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import os
from datetime import datetime
import numpy as np
import torch
from torch import nn
from torchvision import datasets
import torchvision.transforms.v2 as v2

from torch.utils.data import Dataset, DataLoader, random_split
from dl_utils import train_one,test,multiclass_f1_score_aggregation
from torcheval.metrics.functional import (
    multiclass_accuracy
)


BATCH_SIZE = 32
LEARNING_RATE = 1e-1
EPOCHS = 60
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object

print('using DEVICE : ',DEVICE)

torch.manual_seed(53)

train_dir = 'train'
test_dir = 'test'


transforms_train = v2.Compose([
    v2.Resize((48,48)),
    v2.RandomHorizontalFlip(), # data augmentation
    v2.RandomAutocontrast(p = 1.0),
    v2.ToTensor(),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
])

transforms_val = v2.Compose([
    v2.Resize((48,48)),
    v2.ToTensor(),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_ds = datasets.ImageFolder(train_dir, transforms_train)
valid_ds = datasets.ImageFolder(test_dir, transforms_val) # use test folder for validation

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

print('Train dataset size:', len(train_ds))
print('Validation dataset size:', len(valid_ds))

class_names = train_ds.classes
print('Class names:', class_names)




###### MODEL SETUP ######
model = efficientnet_v2_s(weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1)
old_in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Dropout(p = 0.2, inplace = True),
                                 nn.Linear(old_in_features,500),
                                 nn.Dropout(p = 0.15, inplace = True),
                                 nn.Linear(500,len(class_names)))
###### MODEL SETUP ######

   

###### FREEZING LAYER ######
cnt = 0
freeze_threshold = -1
for child in model.children():

  if(cnt < freeze_threshold):
    for param in child.parameters():
      param.requires_grad = False
  else:
    for param in child.parameters():
      param.requires_grad = True

  cnt = cnt + 1
###### FREEZING LAYER ######




writer = SummaryWriter(f'./runs/trainer_{model._get_name()}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')





model.to(DEVICE)
     

loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

     


best_vloss = 100000
for epoch in range(0,EPOCHS):
  train_one(DEVICE,train_dl,model,loss_fn,optimizer,epoch,loss_display_interval = 100)

  train_loss,res_b,res_c = test(DEVICE,train_dl,model,loss_fn,len(class_names))
  train_acc = res_b/len(train_ds)
  train_f1 = multiclass_f1_score_aggregation(res_c,len(class_names))

  valid_loss,res_b,res_c = test(DEVICE,valid_dl,model,loss_fn,len(class_names))
  valid_acc = res_b/len(valid_ds)
  valid_f1 = multiclass_f1_score_aggregation(res_c,len(class_names))

  if valid_loss < best_vloss:
        best_vloss = valid_loss
        torch.save(model.state_dict(), 'model_best_vloss.pth')
        print('Saved best model to model_best_vloss.pth')

  writer.add_scalars(' TLresnet18 Train vs. Valid/loss', 
    {'train':train_loss, 'valid': valid_loss}, 
    epoch)

  writer.add_scalars(
    ' TLresnet18 Performance/acc', 
    {'train':train_acc.item(), 'valid': valid_acc.item()},
    epoch)

  writer.add_scalars(
    ' TLresnet18 Performance/f1', 
    {'train':train_f1, 'valid': valid_f1},
    epoch)


     
