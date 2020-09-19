import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from load_data import load_data

## tensorboard reference:
## https://deeplizard.com/learn/video/pSexXMdruFM
## https://deeplizard.com/learn/video/ycxulUVoNbk

def train(model, loss_func, optimizer, params, tb, data_dir):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloaders, datasizes = load_data(params, 'train', data_dir)

    for epoch in range(params['num_epochs']):

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_func(outputs, labels)

                    if (phase=='train'):
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / datasizes[phase]
            epoch_acc  = running_corrects.double() / datasizes[phase]

            tb.add_scalar(phase+' loss', epoch_loss, epoch)
            tb.add_scalar(phase+' accuracy', epoch_acc, epoch)

            print('Epoch[{}/{}] -- {} Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, params['num_epochs'], phase, epoch_loss, epoch_acc))




#def eval():
