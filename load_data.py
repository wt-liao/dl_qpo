from torchvision import datasets, transforms
from torch.utils.data import random_split
import os
import torch


def load_data(params, phase, data_dir):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if phase == 'train':
        trainset = datasets.ImageFolder(os.path.join(data_dir, phase), transform)

        ##
        train_size = int( 0.95*len(trainset) )
        train_subset, val_subset = random_split( trainset, [train_size, len(trainset)-train_size] )

        trainloader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=int(params["batch_size"]),
            shuffle=True,
            num_workers=4)
        valloader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=int(params["batch_size"]),
            shuffle=True,
            num_workers=4)

        dataloaders = {'train':trainloader, 'val':valloader}
        datasizes   = {'train':train_size,  'val':len(trainset)-train_size}

        return dataloaders, datasizes

    if phase == 'test':
        testset = datasets.ImageFolder(os.path.join(data_dir, phase), transform)

        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=int(params["batch_size"]),
            shuffle=True,
            num_workers=4)
