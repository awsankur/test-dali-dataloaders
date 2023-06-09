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
        files = glob.glob(img_dir + '/**/*.tiff', recursive=True)
     
    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.files)
        
    def __getitem__(self, index):
        # Select sample
        img_path = os.path.join(self.img_dir, self.files[index])

        image_np = imread(img_path)

        #Transform
        image_np = image_np.astype(float)
        # 50x50 5 channel image. Selecting channel 0
        image_np = image_np[:,:,0]

        # Center crop
        image = torch.from_numpy(image_np).permute(2, 0, 1)
        transform = transforms.CenterCrop(0)
        image = transform(image)
        image = image.permute(1, 2, 0)
        image_np = image.detach().cpu().numpy()

        #Normalize
        image_np = utils.normalize_numpy_0_to_1(image_np)

        image = torch.from_numpy(image_np).float().permute(2, 0, 1)
        if self.transform is not None:
            image = self.transform(image)

        return image, index

img_dir = './data/DeepPhenotype_PBMC_ImageSet_YSeverin/Training/'
train_dataset = Dataset(img_dir)


train_loader = DataLoader(train_dataset,shuffle=True,num_workers=0,batch_size=30,pin_memory=True)
start = time.time()
for epoch in range(1, 3):
    for i, data in enumerate(train_loader, 0):
        pass
    end = time.time()
print("Finish with:{} second, num_workers={}".format(end - start, num_workers))


