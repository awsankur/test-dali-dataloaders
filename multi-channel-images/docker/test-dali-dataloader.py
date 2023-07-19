import os
import time
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator



#import nvidia.dali.ops as ops
import numpy as np
import glob
import multiprocessing as mp


IMG_DIR = './data/data/DeepPhenotype_PBMC_ImageSet_YSeverin/Training/'
TRAIN_BS = 30
#NUM_WORKERS = 0
#NUM_IMAGES = 10591
#89572

multichannel_tiff_files = glob.glob(IMG_DIR + '/**/*.tiff', recursive=True)

#import pdb;pdb.set_trace()

# From https://github.com/NVIDIA/DALI/blob/main/dali/test/python/test_pipeline_multichannel.py
@pipeline_def(num_threads=4, device_id=0)
def get_dali_pipeline():

    encoded, label = fn.readers.file(files=multichannel_tiff_files, name="Reader")
    decoded = fn.experimental.decoders.image(encoded, device="mixed", output_type=types.ANY_DATA)

    images = decoded.gpu()
    images = fn.resize(images,resize_y=900, resize_x=300)
    images = fn.crop(images,crop_h=220, crop_w=224,crop_pos_x=0.3, crop_pos_y=0.2)
    #images = fn.transpose(images,perm=(1, 0, 2),transpose_layout=False)
    #images = fn.crop_mirror_normalize(images,std=255., mean=0.,output_layout="HWC",dtype=types.FLOAT)

    return images, label

train_loader = DALIGenericIterator(
    [get_dali_pipeline(batch_size=TRAIN_BS)],
    ['data', 'label'],
    reader_name='Reader'
)

start = time.time()
for epoch in range(1, 3):
    for i, data in enumerate(train_loader, 0):
        pass
    end = time.time()
print("Finish with:{} second".format(end - start))

