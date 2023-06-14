import os
import time
import torch
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.types as types
import numpy as np
import glob
import multiprocessing as mp


IMG_DIR = './data/DeepPhenotype_PBMC_ImageSet_YSeverin/Training/'
#TRAIN_BS = 30
#NUM_WORKERS = 0
#NUM_IMAGES = 10591
#89572

multichannel_tiff_files = glob.glob(IMG_DIR + '/**/*.tiff', recursive=True)


# From https://github.com/NVIDIA/DALI/blob/main/dali/test/python/test_pipeline_multichannel.py
class MultichannelPipeline(Pipeline):
    def __init__(self, device, batch_size, num_threads=1, device_id=0):
        super(MultichannelPipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device

        self.reader = ops.readers.File(files=multichannel_tiff_files)

        decoder_device = 'mixed' if self.device == 'gpu' else 'cpu'
        self.decoder = ops.decoders.Image(device=decoder_device, output_type=types.ANY_DATA)

        self.resize = ops.Resize(device=self.device,
                                 resize_y=900, resize_x=300,
                                 min_filter=types.DALIInterpType.INTERP_LINEAR,
                                 antialias=False)

        self.crop = ops.Crop(device=self.device,
                             crop_h=220, crop_w=224,
                             crop_pos_x=0.3, crop_pos_y=0.2)

        self.transpose = ops.Transpose(device=self.device,
                                       perm=(1, 0, 2),
                                       transpose_layout=False)

        self.cmn = ops.CropMirrorNormalize(device=self.device,
                                           std=255., mean=0.,
                                           output_layout="HWC",
                                           dtype=types.FLOAT)

    def define_graph(self):
        encoded_data, _ = self.reader()
        decoded_data = self.decoder(encoded_data)
        out = decoded_data.gpu() if self.device == 'gpu' else decoded_data
        out = self.resize(out)
        out = self.crop(out)
        out = self.transpose(out)
        out = self.cmn(out)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)






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



