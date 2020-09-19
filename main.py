from itertools import product
from torch.utils.tensorboard import SummaryWriter

from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim

import argparse
from train_eval import train


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## set params grid here
parameters = dict(
    batch_size = [2, 8, 32],
    lr         = [1e-3, 1e-4, 1e-5],
    num_epochs = [10, 20, 40, 80]
)


def main(params, tb, data_dir):

    ## model, loss_func
    model_qpo = models.resnet18(pretrained=True)
    num_ftrs  = model_qpo.fc.in_features
    model_qpo.fc = nn.Linear(num_ftrs, 2)
    model_qpo = model_qpo.to(device)

    ## loss_function
    loss_func = nn.CrossEntropyLoss()

    ## optim
    optimizer = optim.SGD(model_qpo.parameters(), lr=params['lr'], momentum=0.9)

    ## train
    train(model_qpo, loss_func, optimizer, params, tb, data_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data_set')
    args = parser.parse_args()

    param_values = [v for v in parameters.values()]

    #writer = SummaryWriter(logdir='./runs')

    ## pass params to main
    for params in product(*param_values):

        batch_size, lr, num_epochs = params
        comment=f' batch_size={batch_size} lr={lr} num_epochs={num_epochs}'
        tb = SummaryWriter(comment=comment)

        print()
        print(f'Run with batch_size={batch_size}, lr={lr}, num_epochs={num_epochs} --')

        params = {'batch_size':batch_size, 'lr':lr, 'num_epochs':num_epochs}
        main(params, tb, args.data_dir)
