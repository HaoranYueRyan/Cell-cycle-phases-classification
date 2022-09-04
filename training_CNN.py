import torchvision.transforms
from CNN_model import CNN
from torch.utils.data import DataLoader
import torchvision.transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from CCdataset import CCdata
from tqdm import tqdm
import pandas as pd
load_model=False

load_model=False

def save_checkpoint(state,filename='/Users/haoranyue/PycharmProjects/pythonProject/my_CNN100_32_checkpoint.pth.tar'):
    print('=> Saving checkpoint')
    torch.save(state,filename)

def load_checkpoint(checkpoint,model,optimizer):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def train_fn(loader, model, optimizer, criterion, device):
    for batch_idx, (data, targets) in enumerate(tqdm(loader)):
        data = data.to(device=device)
        targets = targets.to(device=device)
        data = data.to(device=device)
        targets = targets.to(device=device)
        #forward
        scores=model(data).to(device=device)
        loss=criterion(scores,targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

         #gradient descent or adam step
        optimizer.step()

def check_accuracy(loader,model,device):
    num_correct=0
    num_samples=0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device=device)
            y=y.to(device=device)
            scores = model(x).to(device=device)
            _,predictions= scores.max(1)
            num_correct += (predictions==y).sum()
            num_samples+= predictions.size(0)
    model.train()
    return num_correct.item()/num_samples

def main():
    in_channel = 1
    num_classes = 4
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 100
    csv_dir = '/Users/haoranyue/Documents/master_project/Output/label10000_true.csv'
    root_dir = '/Users/haoranyue/Documents/master_project/Output/training_set'
    # device using mps
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    datasets = CCdata(csv_file=csv_dir, root_dir=root_dir, transform=torchvision.transforms.Compose(
        [transforms.ToPILImage(), transforms.CenterCrop((60, 60)), transforms.Resize((61, 61)), transforms.ToTensor()]))

    train_size = int(0.8 * len(datasets))
    print(train_size)
    test_size = len(datasets) - train_size
    print(test_size)

    train_set, test_set = torch.utils.data.random_split(datasets, [train_size, test_size])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    model = CNN(in_channels=in_channel, num_classes=num_classes).to(device=device)
    criterion=nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_accurcy=[]
    test_accuracy=[]
    loss_epoch=[]
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device=device)
            targets = targets.to(device=device)
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data).to(device=device)
            loss = criterion(scores, targets)
            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
        loss_epoch.append(loss.item())
        train_accurcy.append(check_accuracy(train_loader, model, device))
        test_accuracy.append(check_accuracy(test_loader, model, device))
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)
    df_loss= pd.DataFrame(loss_epoch, columns=['loss'])
    df_train = pd.DataFrame(train_accurcy, columns=['train_accuracy'])
    df_test = pd.DataFrame(test_accuracy, columns=['test_accuracy'])
    df_train.to_csv('/Users/haoranyue/Documents/master_project/disseration_image/accuracy_epoch/epoch_CNN_100_32_train.csv')
    df_test.to_csv('/Users/haoranyue/Documents/master_project/disseration_image/accuracy_epoch/epoch_CNN_100_32_test.csv')
    df_loss.to_csv('/Users/haoranyue/Documents/master_project/disseration_image/accuracy_epoch/epoch_CNN_100_32_loss.csv')



if __name__=='__main__':
    main()
