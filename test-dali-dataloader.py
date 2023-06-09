import os
import time
import torch
from nvidia.dali import pipeline_def, fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from tifffile import imread
import utils

import multiprocessing as mp


IMG_DIR = './data/DeepPhenotype_PBMC_ImageSet_YSeverin/Training/B'
TRAIN_BS = 30
NUM_WORKERS = 0
NUM_IMAGES = 10591
#89572

@pipeline_def
def simple_pipeline():
    pngs, labels= fn.readers.file(file_root=IMG_DIR,
                                    random_shuffle=True,
                                    name="Reader")
    images = fn.decoders.image(pngs)

    return images, labels

pipe = simple_pipeline(batch_size=TRAIN_BS, num_threads=1, device_id=0)
pipe.build()

images, labels = pipe.run()

import pdb;pdb.set_trace()

train_loader = DALIClassificationIterator([pipe], reader_name='Reader')



start = time.time()
for epoch in range(1, 3):
    for i, data in enumerate(train_loader, 0):
        pass
end = time.time()
print("DALI Finish with:{} second, num_workers={}".format(end - start, NUM_WORKERS))



