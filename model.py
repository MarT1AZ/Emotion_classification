import torch
from torch import nn



class ConvoModule(nn.Module):

  def __init__(self,input_channels,out_channels,kernel_size):
    super(ConvoModule,self).__init__()
    self.conv = nn.Sequential(
                  nn.Conv2d(input_channels,out_channels,kernel_size,padding = 'same'),
                  nn.ReLU(),
                  nn.BatchNorm2d(out_channels),
                  nn.MaxPool2d((2,2)),
                  nn.Dropout2d(0.25)
                  )

  def forward(self,x):
    return self.conv(x)


class FacialNet(nn.Module):

  def __init__(self):
    super(FacialNet,self).__init__()
    self.Conv1 = nn.Conv2d(3,32,(3,3),padding = 'same')
    self.ConvMod1 = ConvoModule(32,64,(3,3))
    self.ConvMod2 = ConvoModule(64,128,(3,3))
    self.ConvMod3 = ConvoModule(128,512,(3,3))
    self.ConvMod4 = ConvoModule(512,512,(3,3))
    self.flatten = nn.Flatten()
    self.fc_module = nn.Sequential(nn.Linear(512 * 3 * 3,256),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(256),
                                   nn.Dropout1d(0.25),
                                   nn.Linear(256,512),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(512),
                                   nn.Linear(512,7))

  def forward(self,x):
    out = self.Conv1(x)
    out = self.ConvMod1(out)
    out = self.ConvMod2(out)
    out = self.ConvMod3(out)
    out = self.ConvMod4(out)
    out = self.flatten(out)
    out = self.fc_module(out)
    return out

  def feature_map_by_layer(self,x):
    feature_maps = []
    out = self.Conv1(x)
    feature_maps.append(out)
    out = self.ConvMod1(out)
    feature_maps.append(out)
    out = self.ConvMod2(out)
    feature_maps.append(out)
    out = self.ConvMod3(out)
    feature_maps.append(out)
    out = self.ConvMod4(out)
    feature_maps.append(out)
    return feature_maps