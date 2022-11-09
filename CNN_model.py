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
        self.conv1= nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False)
        self.nor1=nn.BatchNorm2d(8)
        self.pool_1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool_3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(in_channels = 8, out_channels= 16, kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
        self.nor2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)
        self.nor3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(512, num_classes)
        self.initialize_weights()
        # self.dropout = nn.Dropout(p=0.5)
        # self.upsample1=nn.ConvTranspose2d(in_channels=16,out_channels=8,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        # self.upsample2=nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=(3, 3), stride=(3, 3),
        #                                     padding=(1, 1))

        # self.fc2 = nn.Linear(900, 225)
        # self.fc3 = nn.Linear(225, 84)
        # self.fc4 = nn.Linear(84, num_classes)
        # self.flatten=nn.Flatten(start_dim=1)
        # 16*15*15
    def forward(self,x):

        x=F.relu(self.nor1(self.conv1(x)))

        # print(x.shape,'conv1')
        x=self.pool_1(x)
        # print(x.shape,'pool1')
        x=F.relu(self.nor2(self.conv2(x)))

        x=self.pool_2(x)
        x = F.relu(self.nor3(self.conv3(x)))

        x = self.pool_3(x)




        # print(x,'pool2')
        # y=self.upsample1(x)
        #
        # y=self.upsample2(y)
        # y, x, _=plt.hist(y)
        x= x.reshape(x.shape[0],-1)

        # print(x)
        x = self.fc1(x)
        # x = self.dropout(x)
        # print(x.shape)
        # x = self.fc2(x)
        # # print(x.shape)
        # x = self.dropout(x)
        # x = self.fc3(x)
        # # print(x.shape)
        # x = self.dropout(x)
        # x = self.fc4(x)
        # print(x.shape)
        # print(x.shape,'fc1')
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
    random_data = torch.rand((64, 1, 32, 32,))

    model=CNN(in_channels=1,num_classes=4)
    x=model(random_data)
    print(x.shape)

    # dataset =CCdata(csv_file='/Users/haoranyue/Documents/master_project/Output/label10000_true.csv',
    #    root_dir='/Users/haoranyue/Documents/master_project/Output/training_set/',transform=ToTensorV2())
    # image,label=dataset.__getitem__(1)
    # plt.figure(figsize=(10,10))
    # plt.imshow(image[0])
    # plt.show()


