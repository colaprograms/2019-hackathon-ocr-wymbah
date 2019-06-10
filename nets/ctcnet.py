import torch.nn as nn
import torch, torchvision
from util.chars import nchars

class CTCModel(nn.Module):
  def __init__(self):
    super(CTCModel, self).__init__()
    """
    self.conv = nn.Sequential(
      nn.Conv2d(3, 64, (3, 3), padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, (3, 3), stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, (3, 3), padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 128, (3, 3), stride=2, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 128, (3, 1), padding=(1, 0)),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 128, (3, 1), stride=2, padding=(1, 0)),
      nn.BatchNorm2d(128),
      nn.ReLU(),
    )
    """
    self.makeresnet()
    self.lstm1 = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
    self.layernorm1 = nn.LayerNorm((512,))
    self.dense1 = nn.Sequential(
        nn.Linear(512, 256),
        nn.LayerNorm(256),
        nn.ReLU()
    )
    self.dense2 = nn.Sequential(
        nn.Linear(256, nchars),
    )
    self.dense3 = nn.Sequential(
        nn.Linear(256, nchars),
        nn.Sigmoid()
    )
    self.avgpool = nn.AdaptiveAvgPool2d((1, 32))
  
  def forward(self, z):
    z = z.cuda()
    z = self.conv(z)
    z = self.avgpool(z)
    z = z.squeeze(2).permute(0, 2, 1)
    # batch, seq, channels
    z, _ = self.lstm1(z)
    z = self.layernorm1(z)
    z = self.dense1(z)
    z = self.dense2(z)
    #z = self.dense3(z)
    z = nn.functional.log_softmax(z, dim=2)
    return z
  
  def makeresnet(self):
    resnet = torchvision.models.resnet.resnet34(True)
    resnet.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    
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

class CTCModel4(nn.Module):
  def __init__(self):
    super(CTCModel4, self).__init__()
    """
    self.conv = nn.Sequential(
      nn.Conv2d(3, 64, (3, 3), padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, (3, 3), stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, (3, 3), padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 128, (3, 3), stride=2, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 128, (3, 1), padding=(1, 0)),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 128, (3, 1), stride=2, padding=(1, 0)),
      nn.BatchNorm2d(128),
      nn.ReLU(),
    )
    """
    self.makeresnet()
    self.lstm1 = nn.LSTM(512, 256, batch_first=True, bidirectional=True)
    self.layernorm1 = nn.LayerNorm((512,))
    self.dense1 = nn.Sequential(
        nn.Linear(512, 256),
        nn.LayerNorm(256),
        nn.ReLU()
    )
    self.dense2 = nn.Sequential(
        nn.Linear(256, nchars),
    )
    self.dense3 = nn.Sequential(
        nn.Linear(256, nchars),
        nn.Sigmoid()
    )
    self.avgpool = nn.AdaptiveAvgPool2d((1, 32))
  
  def forward(self, z):
    z = z.cuda()
    z = self.conv(z)
    z = self.avgpool(z)
    z = z.squeeze(2).permute(0, 2, 1)
    # batch, seq, channels
    z, _ = self.lstm1(z)
    z = self.layernorm1(z)
    z = self.dense1(z)
    z = self.dense2(z)
    #z = self.dense3(z)
    z = nn.functional.log_softmax(z, dim=2)
    return z
  
  def makeresnet(self):
    resnet = torchvision.models.resnet.resnet34(True)
    resnet.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    
    "Cut off the last two layers"
    def forward(self, x):
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)

      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.layer4(x)

      return x

    import types
    resnet.forward = types.MethodType(forward, resnet)
    resnet = resnet.cuda()
    #for param in resnet.parameters():
    #  param.requires_grad = False
    self.conv = resnet

class CTCModelBut16(nn.Module):
  def __init__(self):
    super(CTCModelBut16, self).__init__()
    """
    self.conv = nn.Sequential(
      nn.Conv2d(3, 64, (3, 3), padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, (3, 3), stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, (3, 3), padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 128, (3, 3), stride=2, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 128, (3, 1), padding=(1, 0)),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 128, (3, 1), stride=2, padding=(1, 0)),
      nn.BatchNorm2d(128),
      nn.ReLU(),
    )
    """
    self.makeresnet()
    self.lstm1 = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
    self.layernorm1 = nn.LayerNorm((512,))
    self.dense1 = nn.Sequential(
        nn.Linear(512, 256),
        nn.LayerNorm(256),
        nn.ReLU()
    )
    self.dense2 = nn.Sequential(
        nn.Linear(256, nchars),
    )
    self.dense3 = nn.Sequential(
        nn.Linear(256, nchars),
        nn.Sigmoid()
    )
    self.avgpool = nn.AdaptiveAvgPool2d((1, 16))

  def forward(self, z):
    z = z.cuda()
    z = self.conv(z)
    z = self.avgpool(z)
    z = z.squeeze(2).permute(0, 2, 1)
    # batch, seq, channels
    z, _ = self.lstm1(z)
    z = self.layernorm1(z)
    z = self.dense1(z)
    z = self.dense2(z)
    #z = self.dense3(z)
    z = nn.functional.log_softmax(z, dim=2)
    return z

  def makeresnet(self):
    resnet = torchvision.models.resnet.resnet34(True)
    resnet.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    
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

class CTCModelResnetMax(nn.Module):
  def __init__(self):
    super(CTCModel, self).__init__()
    """
    self.conv = nn.Sequential(
      nn.Conv2d(3, 64, (3, 3), padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, (3, 3), stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, (3, 3), padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 128, (3, 3), stride=2, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 128, (3, 1), padding=(1, 0)),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 128, (3, 1), stride=2, padding=(1, 0)),
      nn.BatchNorm2d(128),
      nn.ReLU(),
    )
    """
    self.makeresnet()
    self.lstm1 = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
    self.layernorm1 = nn.LayerNorm((512,))
    self.dense1 = nn.Sequential(
        nn.Linear(512, 256),
        nn.LayerNorm(256),
        nn.ReLU()
    )
    self.dense2 = nn.Sequential(
        nn.Linear(256, nchars),
    )
    self.dense3 = nn.Sequential(
        nn.Linear(256, nchars),
        nn.Sigmoid()
    )
    self.avgpool = nn.AdaptiveAvgPool2d((1, 32))
  
  def forward(self, z):
    z = z.cuda()
    z = self.conv(z)
    z = self.avgpool(z)
    z = z.squeeze(2).permute(0, 2, 1)
    # batch, seq, channels
    z, _ = self.lstm1(z)
    z = self.layernorm1(z)
    z = self.dense1(z)
    z = self.dense2(z)
    #z = self.dense3(z)
    #print(z[:, 0, 0])
    #print(z.shape)
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

class CTCModel___(nn.Module):
  def __init__(self):
    super(CTCModel, self).__init__()
    """
    self.conv = nn.Sequential(
      nn.Conv2d(3, 64, (3, 3), padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, (3, 3), stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, (3, 3), padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 128, (3, 3), stride=2, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 128, (3, 1), padding=(1, 0)),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 128, (3, 1), stride=2, padding=(1, 0)),
      nn.BatchNorm2d(128),
      nn.ReLU(),
    )
    """
    self.makeresnet()
    self.lstm1 = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
    self.layernorm1 = nn.LayerNorm((512,))
    self.dense1 = nn.Sequential(
        nn.Linear(512, 256),
        nn.LayerNorm(256),
        nn.ReLU()
    )
    self.dense2 = nn.Sequential(
        nn.Linear(256, nchars),
    )
    self.avgpool = nn.AdaptiveAvgPool2d((1, 32))
  
  def forward(self, z):
    z = z.cuda()
    z = self.conv(z)
    z = self.avgpool(z)
    z = z.squeeze(2).permute(0, 2, 1)
    # batch, seq, channels
    z, _ = self.lstm1(z)
    z = self.layernorm1(z)
    z = self.dense1(z)
    z = self.dense2(z)
    #z = self.dense3(z)
    z = nn.functional.log_softmax(z, dim=2)
    return z
  
  def makeresnet(self):
    resnet = torchvision.models.resnet.resnet34(True)
    resnet.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    
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

class CTCModel512(nn.Module):
  def __init__(self):
    super(CTCModel, self).__init__()
    self.makeresnet()
    self.lstm1 = nn.LSTM(256, 512, batch_first=True, bidirectional=True)
    self.layernorm1 = nn.LayerNorm((1024,))
    self.dense1 = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.LayerNorm(1024),
        nn.ReLU()
    )
    self.dense2 = nn.Sequential(
        nn.Linear(1024, nchars),
    )
    self.avgpool = nn.AdaptiveAvgPool2d((1, 32))
  
  def forward(self, z):
    z = z.cuda()
    z = self.conv(z)
    z = self.avgpool(z)
    z = z.squeeze(2).permute(0, 2, 1)
    # batch, seq, channels
    z, _ = self.lstm1(z)
    z = self.layernorm1(z)
    z = self.dense1(z)
    z = self.dense2(z)
    #z = self.dense3(z)
    z = nn.functional.log_softmax(z, dim=2)
    return z
  
  def makeresnet(self):
    resnet = torchvision.models.resnet.resnet34(True)
    resnet.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    
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
