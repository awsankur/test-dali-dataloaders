# Accelerate Distributed Training of PyTorch Deep Learning Models with Nvidia DALI dataloaders
In this repo, we will show some results benchmarking data loading and pre-processing speedups obtained with Dataloaders created with [NVIDIA DALI](https://developer.nvidia.com/dali). For distributed training applications, especially those that need to read images and pre-process them, data loading could end up being a bottleneck as traditionally these loading and pre-processing tasks are CPU bound. To this end, we can leverage multi-processing by providing [`num_workers`](https://pytorch.org/docs/stable/data.html) in a PyTorch DataLoader but often that is not enough. With NVIDIA DALI, one can use the existing GPUs to do the data loading and pre-processing of the images that significantly accerelerates data loading, preprocessing and thereby distributed training. Next we will show that for a distributed training example with multi-channel images, average time per epoch can be reduced by upto 95%!

## Data Source
We will use a deep phenotyping multi-channel cell images open sourced [here](https://www.research-collection.ethz.ch/handle/20.500.11850/343106). We will use ~90K images given in the Training set in this data. You can download this data by executing [`./1-get-data.sh`](https://github.com/awsankur/test-dali-dataloaders/blob/main/multi-channel-images/1-get-data.sh). You can run the script [./count_number_of_training_images.sh](https://github.com/awsankur/test-dali-dataloaders/blob/main/multi-channel-images/count_number_of_training_images.sh) to count the number of training images.

## Installing DALI
The latest version of DALI is included in the latest PyTorch [NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch). In this work we use an image built from `nvcr.io/nvidia/pytorch:23.06-py3`.

## 



## PyTorch DataLoader
`train_loader = DataLoader(train_dataset,shuffle=True,num_workers=0,batch_size=30,pin_memory=True)`

90K images, 3 epochs \n

```
Finish with:112.29331731796265 second, num_workers=0,batch_size = 30
Finish with:60.46501398086548 second, num_workers=2 batch_size = 30
Finish with:31.465123891830444 second, num_workers=4 batch_size = 30
Finish with:17.2755343914032 second, num_workers=8 batch_size = 30
Finish with:12.683705568313599 second, num_workers=16, batch_size = 30
Finish with:10.232728004455566 second, num_workers=32, batch_size = 30
Finish with:8.886242866516113 second, num_workers=32, batch_size = 60
Finish with:8.396265268325806 second, num_workers=32, batch_size = 96
```

