import matplotlib.pyplot as plt
import torch
import torchvision
from dapi_tubulins import channelsDataset
from torch.utils.data import DataLoader

def save_checkpoint (state,filename='my_checkpoint_unetboneEfficient.pth.tar'):
    print('=> Saving checkpoint')
    torch.save(state,filename)

def load_checkpoint(checkpoint,model,optimizer):
    print('=> Loadinig checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory= True,

):
    train_ds = channelsDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,

    )

    val_ds = channelsDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,

    )

    return train_loader, val_loader
accuracy_value = []


def check_accuracy(loader,model,device='mps'):
    num_correct= 0
    num_pixels =0
    dice_score =0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x=x.to(device)
            y=y.to(device)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f'GOT {num_correct.item()}/{num_pixels} with acc {(num_correct.item())/num_pixels*100:.2f}'
    )

    print(
        f'Dice score: {dice_score/len(loader)}'
    )
    model.train()
    return (num_correct.item())/num_pixels*100



def save_predictions_as_imgs(
        loader,model,folder='saved_images/',device='mps'
):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x=x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds= (preds>0.5).float()
            torchvision.utils.save_image(
            preds*255,f'{folder}/pred_{idx}.png'
            )
        torchvision.utils.save_image(y*255,f"{folder}{idx}.tif")
    model.train()
