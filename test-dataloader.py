import os
import time
import glob
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tifffile import imread
import utils

import multiprocessing as mp


class Dataset(torch.utils.data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, img_dir):
    
        #'Initialization'
        self.img_dir = img_dir
        self.files = glob.glob(img_dir + '/**/*.tiff', recursive=True)
     
    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.files)
        
    def __getitem__(self, index):
        # Select sample
        image_np = imread(self.files[index])

        #Transform
        image_np = image_np.astype(float)
        image = torch.from_numpy(image_np)
        # 50x50 5 channel image. Selecting channel 0
        #image_np = image_np[:,:,0]

        #transform = transforms.Compose([
        #        transforms.Resize(size=(220,224)),
        #        transforms.CenterCrop(0)
        #        ])

        
        #image = transform(image)

        return image, index

img_dir = './data/data/DeepPhenotype_PBMC_ImageSet_YSeverin/Training/'
train_dataset = Dataset(img_dir)

num_workers = 0
train_loader = DataLoader(train_dataset,shuffle=True,num_workers=num_workers,batch_size=30,pin_memory=True)
start = time.time()
for epoch in range(1, 3):
    for i, data in enumerate(train_loader, 0):
        pass
    end = time.time()
print("Finish with:{} second, num_workers={}".format(end - start, num_workers))


