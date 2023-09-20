import torch
from torch import nn
from torchvision.models import vgg19
# Import config file if not using ipynb 
import config

class VGGLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.vgg = vgg19(pretrained = True).features[:36].eval().to(config.DEVICE)
    self.loss = nn.MSELoss()
    
    for param in self.vgg.parameters():
      param.requires_grad = False



  def forward(self,input,target):
    # Input is low res
    # Output is high res image

    vggInputFeatures = self.vgg(input)
    vggTargetFeatures = self.vgg(target)
    return self.loss(vggInputFeatures,vggTargetFeatures)

