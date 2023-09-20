
import torch 
from torch import nn

class ConvBlock(nn.Module):
  # Conv -> BatchNorm -> Leaky / PReLU
  def __init__(self,
               inChannels,
               outChannels,
               discriminator = False,
               useActivation = True,
               useBatchNorm = False, 
               **kwargs,):
    super().__init__()
    self.useActivation = useActivation
    self.cnn = nn.Conv2d(inChannels,outChannels,**kwargs,bias = not useBatchNorm)
    self.bn = nn.BatchNorm2d(outChannels) if useBatchNorm else nn.Identity()
    self.activation = (
        nn.LeakyReLU(0.2, inplace = True)
        if Discriminator
        else nn.PReLU(num_parameters = outChannels)
    )

  def forward(self,x):
      return self.activation(self.bn(self.cnn(x))) if self.useActivation else self.bn(self.cnn(x))

class UpsampleBlock(nn.Module):
  # 
  def __init__(self,inChannels,scale):
    super().__init__()
    self.conv = nn.Conv2d(inChannels,inChannels * scale **2 ,3,1,1)
    # upsample the image so that h , w transform to h*2 and w*2
    self.pixelShuffle = nn.PixelShuffle(scale) 
    self.activation = nn.PReLU(num_parameters=inChannels)

  def forward(self,x):
    return self.activation(self.pixelShuffle(self.conv(x)))

class ResidualBlock(nn.Module):
  def __init__(self,inChannels):
    super().__init__()
    self.block1 = ConvBlock(inChannels,
                            inChannels,
                            kernel_size = 3,
                            stride = 1,
                            padding=1)
    
    self.block2 = ConvBlock(inChannels,
                            inChannels,
                            kernel_size = 3,
                            stride = 1,
                            padding=1,
                            useActivation= False)
  def forward(self,x):
    out = self.block1(x)
    out = self.block2(out)
    # add x for residual
    return out + x

class Generator(nn.Module):
  def __init__(self,inChannels = 3,numChannels = 64,numBlocks = 16):
    super().__init__()
    self.initial = ConvBlock(inChannels,
                             numChannels,
                             kernel_size = 9, 
                             stride = 1, 
                             padding = 4, 
                             useBatchNorm=False)
    self.residuals = nn.Sequential(
        *[ResidualBlock(numChannels) for _ in range(numBlocks)]
    )
    self.convblock = ConvBlock(numChannels,numChannels, kernel_size = 3, stride = 1, padding = 1, useActivation = False)
    self.upsamples = nn.Sequential(UpsampleBlock(numChannels,2),UpsampleBlock(numChannels,2))
    self.final = nn.Conv2d(numChannels,inChannels,kernel_size = 9, stride = 1,padding = 4)
  def forward(self,x):
    initial = self.initial(x)
    x = self.residuals(initial)
    x = self.convblock(x) + initial
    x = self.upsamples(x)
    return torch.tanh(self.final(x))


class Discriminator(nn.Module):
  def __init__(self,inChannels = 3, features = [64,64,128,128,256,256,512,512]):
    super().__init__()
    blocks = []
    for idx,feature in enumerate(features):
      blocks.append(ConvBlock(
          inChannels,feature,
          kernel_size = 3, 
          stride = 1 + idx %2, #Stride is 1 for odd number blocks, 2 for even
          padding = 1,
          useActivation= True,
          useBatchNorm=False if idx == 0 else True,
          discriminator=True

      ))
      inChannels = feature

      self.blocks = nn.Sequential(*blocks)
      self.classifier = nn.Sequential(
          nn.AdaptiveAvgPool2d((6,6)),
          nn.Flatten(),
          nn.Linear(512*6*6,1024),
          nn.LeakyReLU(0.2,inplace = True),
          nn.Linear(1024,1),
      )
  def forward(self,x):
    x = self.blocks(x)
    return self.classifier(x)
# Testing out if the shapes and dimensions are okay
def test():
  # feed in low res 24 x 24 image, should scale to 96 (scale by 4)
  lowResolution = 24
  with torch.cuda.amp.autocast():
    x = torch.randn((5,3,lowResolution,lowResolution))
    generator = Generator()
    generatorResult = generator(x)
    discriminator = Discriminator()
    discriminatorResult = discriminator(generatorResult)

    print(generatorResult.shape)
    print(discriminatorResult.shape)
    