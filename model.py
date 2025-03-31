
import torch
from torch import nn
import torch.nn.functional as F



class FacialNet(nn.Module):


    def __init__(self,num_class):

        super().__init__()
        # input 48 by 48
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = torch.nn.Conv2d(3,12,kernel_size = 2)
        self.conv2 = torch.nn.Conv2d(12,24,kernel_size = 2)
        self.conv3 = torch.nn.Conv2d(24,36,kernel_size = 2)
        self.fc = nn.Sequential(nn.Linear(900,600),
                                nn.Relu(),
                                nn.Linear(600,100),
                                nn.Relu(),
                                nn.linear(100,num_class))


    def forward(self,x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = self.pool(F.relu(self.conv3(out)))
        out = torch.flatten(out, 1) # flatten all dimensions except batch
        out = self.fc(out)
        return out
