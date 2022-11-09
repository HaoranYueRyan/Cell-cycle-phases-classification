import torchvision.transforms

import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from CCdataset import CCdata
from CNN_model import CNN
from Autoencoder_model import Autoencoderv3
from ResNet_model import ResNet50
from ResNet_model import ResNet101
from ResNet_model import ResNet152
from EfficientNet_model import EfficientNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
load_model=False
import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from torchvision.models import vgg11
from tqdm import tqdm



def save_checkpoint(state,filename='/Users/haoranyue/PycharmProjects/pythonProject/vgg11__09_checkpoint_pth.tar'):
    print('=> Saving checkpoint')
    torch.save(state,filename)

def load_checkpoint(checkpoint,model,optimizer):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])



in_channel=1
num_classes=4
learning_rate=0.001
batch_size=32
num_epochs=10

#device using mps
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")






def check_accuracy(loader,model,device):
    num_correct=0
    num_samples=0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device=device)
            y=y[0].to(device=device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions==y).sum()
            num_samples+= predictions.size(0)


    model.train()
    return num_correct.item()/num_samples

def main():
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 100
    csv_dir = '/Users/haoranyue/Documents/master_project/df_label_11_06_4.csv'
    root_dir = '/Users/haoranyue/Documents/master_project/output_17_3'
    # device using mps
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    transform = A.Compose(
        [
            A.Resize(32, 32,),
            # A.CenterCrop(60, 60,),
            # A.Resize(32, 32),





            A.Rotate(limit=40, p=0.6),
            A.HorizontalFlip(p=0.6),
            A.RandomBrightnessContrast(p=0.6),


            # A.RandomBrightnessContrast(p=0.8),
            A.VerticalFlip(p=0.6),
            A.OneOf(
                [
                    # A.Blur(blur_limit=3, p=0.8),
                    # A.ColorJitter(p=0.5),

                ], p=1.0
            ),
            ToTensorV2(),
        ]
    )


    datasets= CCdata(csv_file=csv_dir, root_dir=root_dir, transform=transform)

    # datasets = CCdata(csv_file=csv_dir, root_dir=root_dir, transform=torchvision.transforms.Compose(
    #     [transforms.ToPILImage(), transforms.CenterCrop((60, 60)), transforms.Resize((61, 61)), transforms.ToTensor()]))

    train_size = int(0.8 * len(datasets))
    print(train_size)
    test_size = len(datasets) - train_size
    print(test_size)

    train_set, test_set = torch.utils.data.random_split(datasets, [train_size, test_size])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


    model=vgg11(num_classes=4).to(device=device)
    model.features[0]=nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model=model.to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        load_checkpoint(torch.load('/Users/haoranyue/PycharmProjects/pythonProject/my_Efficientnetb0_10_32_checkpoint_try_08_22.pth.tar'),model, optimizer)
        model.to(device=device)

    train_accurcy = []
    test_accuracy = []
    loss_epoch = []
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device=device)
            targets = targets[0].to(device=device)

            # forward
            scores = model(data).to(device=device)

            loss = criterion(scores, targets)
            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
        loss_epoch.append(loss.item())
        print(loss_epoch)
        train_accurcy.append(check_accuracy(train_loader, model, device))
        print(train_accurcy)
        test_accuracy.append(check_accuracy(test_loader, model, device))
        print(test_accuracy)
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)
    df_loss = pd.DataFrame(loss_epoch, columns=['loss'])
    df_train = pd.DataFrame(train_accurcy, columns=['train_accuracy'])
    df_test = pd.DataFrame(test_accuracy, columns=['test_accuracy'])
    df_train.to_csv('/Users/haoranyue/Documents/master_project/disseration_image/accuracy_epoch/epoch_v11_100_32_train_09.csv')
    df_test.to_csv('/Users/haoranyue/Documents/master_project/disseration_image/accuracy_epoch/epoch_v11_100_32_test_09.csv')
    df_loss.to_csv('/Users/haoranyue/Documents/master_project/disseration_image/accuracy_epoch/epoch_v11_100_32_loss_09.csv')
#
if __name__=='__main__':

    main()

