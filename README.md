# test-dali-dataloaders
Benchmark DALI dataloader speedups

## PyTorch DataLoader
`train_loader = DataLoader(train_dataset,shuffle=True,num_workers=0,batch_size=30,pin_memory=True)`

90K images, 3 epochs \n

`Finish with:112.29331731796265 second, num_workers=0`
