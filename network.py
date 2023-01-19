import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torch.autograd import Variable
import pytorch_lightning as pl
import math
import pdb
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict
import copy
from torchmetrics import Accuracy

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        try:
            nn.init.kaiming_uniform_(m.weight) # use ubiform init to weight make activate function not full and non-zero
            nn.init_zeros_(m.bias)
        except:
            pass
    elif classname.find('BatcNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        try:
            nn.init.zeros_(m.bias)
        except AttributeError:
            pass

class AutoEncoder(pl.LightningModule):
    def __init__(self, type="linear"):
        super(AutoEncoder, self).__init__()
        if type== "linear":
            self.encoder = nn.Sequential(
                nn.Conv2d(3,64,5,1), #224 -4
                nn.ReLU(),
                nn.MaxPool2d(2), #110
                nn.Conv2d(64,64,5,1), #110 -4 =106
                nn.ReLU(),
                nn.MaxPool2d(2), #106/2=53
                nn.Flatten(),
                nn.Linear(53*53*64,64),
                nn.ReLU(),
                nn.Linear(64,3)
            )
            self.deconder = nn.Sequential(
                nn.Linear(3,64),
                nn.ReLU(),
                nn.Linear(64,224*224*3)
            )
        
        else :
            self.encoder = weightNorm(nn.Sequential(
                nn.Linear(224*224*3,64),
                nn.ReLU(),
                nn.Linear(64,3)
            ))
            self.deconder = weightNorm(nn.Sequential(
                nn.Linear(3,64),
                nn.ReLU(),
                nn.Linear(64,224*224*3)
            ))
        self.encoder.apply(init_weight)
        self.deconder.apply(init_weight)
        
    
    def forward(self,x):
        embedding = self.encoder(x)
        debedding = self.deconder(embedding)
        return debedding
    
    # def configure_optimizers(self):
    #     # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    #     optimizer = torch.optim.SGD([{
    #     'params': self.encoder.parameters(),
    #     'lr': 1e-3
    #     }, 
    #     {
    #     'params': self.deconder.parameters(),
    #     'lr': (1e-3) * 10 
    #     }, ],
    #     momentum=0.9,
    #     weight_decay=0.0005,
    #     nesterov=True)

    #     return optimizer
    
    # def training_step(self, train_batch, batch_idx):
    #     x, y = train_batch
    #     z = self.encoder(x)
    #     x_hat = self.deconder(z)   
    #     x_hat = torch.reshape(x_hat,(x.shape[0],3, 224,224))
        
      

    #     loss = F.mse_loss(x_hat ,x)
    #     self.log('train_loss', loss)
    #     return loss 

    # def validation_step(self, val_batch,batch_idx):
    #     x, y = val_batch
    #     z = self.encoder(x)
    #     x_hat = self.deconder(z)
    #     x_hat = torch.reshape(x_hat,(x.shape[0],3, 224,224))
    #     x_z = x_hat.cpu().detach().numpy()
    #     x_z = x_z.reshape(x_z.shape[0],3,224,224)
    #     loss = F.mse_loss(x_hat,x)
    #     self.log('val_loss', loss)
    #     # print('val_loss', loss)
    #     return loss 
    
    # def test_step(self, val_batch, batch_idx):
    #     x, y = val_batch
    #     z = self.encoder(x)
    #     x_hat = self.deconder(z)
    #     x_hat = torch.reshape(x_hat,(x.shape[0],3, 224,224))
    #     x_z = x_hat.cpu().detach().numpy()
    #     x_z = x_z.reshape(x_z.shape[0],224,224,3)
    #     show_images(x_z)
    #     plt.show()
    #     loss = F.mse_loss(x_hat,x)
    #     self.log("test_loss", loss)
    #     return loss 



class feat_classifier(nn.Module):
    def __init__(self, type="linear", class_num=2, bottleneck_dim=256):
        super(feat_classifier, self).__init__()
        if type == "linear":
            # self.fc = nn.Linear(bottleneck_dim, class_num, bias=False)
            self.fc1 = nn.Linear(bottleneck_dim, 128, bias=False)
            self.fc2 = nn.Linear(128, class_num)
            # self.fcclassifier = nn.Sequential(self.fc1,
            #                                  nn.ReLU(),
            #                                  nn.Droput(0.2),
            #                                  self.fc2)            
        else:
            self.fc1 = weightNorm(nn.Linear(bottleneck_dim, 128,),
                                 name="weight")
            self.fc2 =  weightNorm(nn.Linear(128, class_num))
            # self.fcclassifier = nn.Sequential(self.fc1,
            #                                  nn.ReLU(),
            #                                  nn.Droput(0.2),
            #                                  self.fc2)
        # self.fcclassifier.apply(init_weight)
    
    def forward(self, x):
        # x = self.fc(x)  #/0.05
        # return x
        # out = self.fcclassifier(x)
        out = self.fc1(x)
        out = self.fc2(out)
        
        return out




class ResNet_FE(nn.Module):
    def __init__(self):
        super().__init__()
        model_resnet = torchvision.models.resnet50(True)
        self.conv1 = model_resnet.conv1
        self.bn1  = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu,
                                            self.maxpool, self.layer1,
                                            self.layer2, self.layer3,
                                            self.layer4, self.avgpool)
        self.bottle = nn.Linear(2048,256)
        self.bn = nn.BatchNorm1d(256)
    
    def forward(self, x):
        out = self.feature_layers(x)
        out = out.view(out.size(0),-1)
        out = self.bn(self.bottle(out))
        
        return out






