from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import torch
from torch import nn
import torch.nn.functional as F
from model import FacialNet
import numpy as np


test_tensor = torch.Tensor(np.ndarray(shape = (100,3,48,48)))


model = FacialNet(6)
print(model(test_tensor).shape)

