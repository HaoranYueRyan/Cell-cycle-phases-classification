import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from CCdataset import CCdata
from albumentations.pytorch import ToTensorV2

class CNN(nn.Module):
    def __init__(self,in_channels, num_classes):
        super(CNN,self).__init__()
        # using channels of 16, 64, 128, 256, 256 to get the features from original images
        self.features=nn.Sequential(
              #channles from inpurt to 16
              nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False),
              nn.BatchNorm2d(16),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
              #channles from 16 to 64
              nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False),
              nn.BatchNorm2d(64),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
              #channles from 64 to 128
              nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False),
              nn.BatchNorm2d(128),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
              #channles from 128 to 256
              nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False),
              nn.BatchNorm2d(256),
              nn.ReLU(inplace=True),
              #channles from 256 to 512
              nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
              nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False),
              nn.BatchNorm2d(512),
              nn.ReLU(inplace=True),
              #channles from 512 to 512
              nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False),
              nn.BatchNorm2d(512),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
              #flatten the finally channels
              nn.Flatten()
        )

        # using nn.linear function to classifier the num_classes
        self.classifier=nn.Sequential(
                         # nn.Linear(in_features=4096, out_features=4096, bias=True),
                         # nn.ReLU(inplace=True),
                         # nn.Dropout(p=0.5, inplace=False),
                         # nn.Linear(in_features=4096, out_features=4096, bias=True),
                         # nn.ReLU(inplace=True),
                         # nn.Dropout(p=0.5, inplace=False),
                         nn.Linear(in_features=512, out_features=num_classes, bias=True),
                       )

        # init the model weights
        self.initialize_weights()


    def forward(self,x):
        x=self.features(x)
        # x= x.reshape(x.shape[0],-1)
        x=self.classifier(x)
        return x

    def initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


if __name__=='__main__':
    random_data = torch.rand((64, 3, 64, 64,))

    model=CNN(in_channels=3,num_classes=4)
    x=model(random_data)
    print(x.shape)


