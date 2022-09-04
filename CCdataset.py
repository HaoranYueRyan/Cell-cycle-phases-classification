import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from natsort import natsorted
from torchvision.utils import save_image
import torchvision.transforms as transforms
import numpy as np

class CCdata(Dataset):
    def __init__(self, csv_file, root_dir,transform=None):
        self.annotations=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir,(self.annotations.iloc[index,0]+'.tif'))
        # print(img_path)

        image=Image.open(img_path)
        image=np.float32(image)
        image = image / np.amax(image)
        image= np.clip(image, 0, 1)

        # img=np.float32(image)
        # plt.Figure(figsize=(10,10))
        # plt.imshow(image)
        # plt.show()

        y_lable= torch.tensor(int(self.annotations.iloc[index,1])),

        # print((np.float32(image) / 255)
        if self.transform:
            # image = self.transform(image=image)
            augmentations= self.transform(image=image)
            image=augmentations['image']
            # plt.Figure(figsize=(10, 10))
            # plt.imshow(image[0])
            # plt.show()
            # save_image(image,'/Users/haoranyue/Documents/master_project/Output/transofrom_img/imaage' + str(10000) + '.png'
            #            )

        return (image,y_lable)

if __name__=='__main__':
    dataset =CCdata(csv_file='/Users/haoranyue/Documents/master_project/Output/check_transform.csv',
       root_dir='/Users/haoranyue/Documents/master_project/Output/training_set/',transform=transforms.Compose(
    [transforms.ToPILImage(),
     transforms.CenterCrop((60, 60)),
     transforms.ToTensor(),

     ]
        ))
    dataset.__getitem__(1)
