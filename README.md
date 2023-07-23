# Accelerate Distributed Training of PyTorch Deep Learning Models with Nvidia DALI dataloaders
In this repo, we will show some results benchmarking data loading and pre-processing speedups obtained with Dataloaders created with [NVIDIA DALI](https://developer.nvidia.com/dali)

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

