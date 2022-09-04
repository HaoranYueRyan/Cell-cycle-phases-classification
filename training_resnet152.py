import torchvision.transforms
import matplotlib.pyplot as plt
from CCdataset import CCdata
from CNN_model import CNN
from ResNet_model import ResNet50
from ResNet_model import ResNet101
from ResNet_model import ResNet152
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from CCdataset import CCdata
import pandas as pd
from tqdm import tqdm
load_model=False

def save_checkpoint(state,filename='/Users/haoranyue/PycharmProjects/pythonProject/my_resnet152_50_32_checkpoint.pth.tar'):
    print('=> Saving checkpoint')
    torch.save(state,filename)

def load_checkpoint(checkpoint,model,optimizer):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()

in_channel=1
num_classes=4
learning_rate=0.001
batch_size=32
num_epochs=50
csv_dir='/Users/haoranyue/Documents/master_project/Output/label10000_true.csv'
root_dir='/Users/haoranyue/Documents/master_project/Output/ALL'
#device using mps
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")




datasets = CCdata(csv_file=csv_dir,root_dir=root_dir,transform=torchvision.transforms.Compose([transforms.ToPILImage(),transforms.CenterCrop((60,60)),transforms.Resize((61,61)),transforms.ToTensor()]))

train_size = int(0.7*len(datasets))
print(train_size)
test_size = len(datasets) - train_size
print(test_size)

train_set,test_set=torch.utils.data.random_split(datasets,[train_size ,test_size])
train_loader=DataLoader(dataset=train_set,batch_size=batch_size,shuffle= True)
test_loader=DataLoader(dataset=test_set,batch_size=batch_size,shuffle= True)

model= ResNet152(img_channels=1,num_classes=4).to(device=device)

# img.type(torch.cuda.FloatTensor)
#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'),model,optimizer)

# Train Network
loss_value = []
for epoch in range(num_epochs):



    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):

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

    if epoch==(num_epochs-1):

        checkpoint={'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()}

        save_checkpoint(checkpoint)

    loss_value.append(loss.item())

f_epoch = pd.DataFrame(loss_value, columns=['loss'])
df_epoch.to_csv('/Users/haoranyue/Documents/master_project/Output/epoch_ResNet152_50_32.csv')

for index,i  in enumerate(loss_value):
    print(f"epoch {index} loss on training set: {i}")

# Check accuracy on trainning to see how good our model is
def check_accuracy(loader,model):
    num_correct=0
    num_samples=0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device=device)
            y=y.to(device=device)


            scores = model(x)

            _,predictions= scores.max(1)

            num_correct += (predictions==y).sum()

            print(num_correct)

            num_samples+= predictions.size(0)


    model.train()
    return num_correct.item()/num_samples

print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
