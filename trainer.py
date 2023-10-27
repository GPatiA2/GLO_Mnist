import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

class Callback():

    def __init__(self, func, interval):
        self.func = func
        self.interval = interval

class Initializer():

    @abstractmethod
    def initialize(self, dataset): ...


class RandomCodeInitializer(Initializer):

    def __init__(self, seed, dim):
        self.seed = seed
        self.dim  = dim

    def initialize(self, dataset):
        torch.manual_seed(self.seed)
        return torch.randn(len(dataset), self.dim, requires_grad=True)


class GLO_Trainer():

    def __init__(self, model, dataset, dataloader, criterion, initializer):

        self.model = model
        self.dataset = dataset
        self.dataloader = dataloader
        self.criterion = criterion
        self.code_initializer = initializer

        self.z_codes = self.code_initializer.initialize(self.dataset)

        self.train_losses = []

        self.batch_size = self.dataloader.batch_size     

        self.num_epoch = 0   

    def set_optimizer(self, optimizer, lr, b1, b2):
        p = [
            {'params': self.model.parameters() , 'lr':lr},
            {'params': self.z_codes, 'lr':lr}
        ]
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(p, lr=lr, betas=(b1, b2))
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(p, lr=lr, momentum=b1)
        else:
            raise ValueError('Optimizer not supported')

    def get_codes(self, idx):

        codes = []
        for i in idx:
            codes.append(self.z_codes[i])
        
        return torch.stack(codes)
    
    def train_for_epochs(self, epochs, callbacks):

        self.model = self.model.to('cuda')
        self.z_codes = self.z_codes.to('cuda')

        for epoch in range(epochs):
            
            self.epoch()

            for cb in callbacks:
                if epoch % cb.interval == 0:
                    cb.func(self)

            self.num_epoch += 1

    def save_checkpoint(self, path):

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'z_codes': self.z_codes
        }, path + '/checkpoint_' + str(self.num_epoch) + '.pt')

    def save_losses(self, path):

        x = [i for i in range(len(self.train_losses))]
        y = self.train_losses

        plt.plot(x, y)
        plt.savefig(path + '/losses_' + str(self.num_epoch) + '.png')
    

    def epoch(self):

        for batch in tqdm(self.dataloader):

            self.optimizer.zero_grad()

            image = batch[0]
            image = [torch.from_numpy(np.float32(i)) for i in image]
            image = torch.stack(image, dim = -1)
            image = image.to('cuda')
            idx = batch[1]

            code = self.get_codes(idx)

            reconstructed = torch.squeeze(self.model(code))

            loss = torch.sum(torch.abs(image - reconstructed))

            loss.backward()

            self.optimizer.step()

            code = nn.functional.normalize(code, dim=0, p = 2)

            # for i in range(len(idx)):
            #     self.z_codes[idx[i]] = code[i]

            self.train_losses.append(loss.detach().to('cpu').numpy())
