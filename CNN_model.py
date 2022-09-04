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
    def __init__(self,in_channels=1, num_classes = 4):
        super(CNN,self).__init__()
        self.conv1= nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels= 16, kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.fc1 = nn.Linear(16*15*15, num_classes)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        # print(x.shape,'conv1')
        x=self.pool(x)
        # print(x.shape,'pool1')
        x=F.relu(self.conv2(x))
        # print(x.shape,'conv2')
        x=self.pool(x)
        # print(x.shape,'pool2')
        x= x.reshape(x.shape[0],-1)
        # print(x.shape,'conv2')
        x = self.fc1(x)
        # print(x.shape,'fc1')
        return x

if __name__=='__main__':
    dataset =CCdata(csv_file='/Users/haoranyue/Documents/master_project/Output/label10000_true.csv',
       root_dir='/Users/haoranyue/Documents/master_project/Output/training_set/',transform=ToTensorV2())
    image,label=dataset.__getitem__(1)
    plt.figure(figsize=(10,10))
    plt.imshow(image[0])
    plt.show()


