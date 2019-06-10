import torch.nn as nn
import torch, torchvision
from util.chars import nchars

class CTCModel(nn.Module):
  def __init__(self):
    super(CTCModel, self).__init__()
    self.makeresnet()
    self.lstm1 = nn.LSTM(256, 512, batch_first=True)
    self.dense = nn.Sequential(
        nn.LayerNorm((512,)),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 64),
        nn.LayerNorm((64,)),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, nchars)
    )
    self.avgpool = nn.AdaptiveAvgPool2d((1, 32))
  
  def forward(self, z):
    z = z.cuda()
    z = self.conv(z)
    z = self.avgpool(z)
    z = z.squeeze(2).permute(0, 2, 1)
    z, _ = self.lstm1(z)
    z = self.dense(z)
    z = nn.functional.log_softmax(z, dim=2)
    return z
  
  def makeresnet(self):
    resnet = torchvision.models.resnet.resnet34(True)
    
    "Cut off the last two layers"
    def forward(self, x):
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)

      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      #x = self.layer4(x)

      return x

    import types
    resnet.forward = types.MethodType(forward, resnet)
    resnet = resnet.cuda()
    #for param in resnet.parameters():
    #  param.requires_grad = False
    self.conv = resnet
