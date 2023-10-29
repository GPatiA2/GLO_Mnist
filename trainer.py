import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import torchvision
import cv2

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
        return torch.randn(len(dataset), self.dim)


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
        ]
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(p, lr=lr, betas=(b1, b2))
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(p, lr=lr, momentum=b1)
        else:
            raise ValueError('Optimizer ' + optimizer  + ' not supported')

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
        }, path + '/checkpoints/' + str(self.num_epoch) + '.pt')

    def save_losses(self, path):

        x = [i for i in range(len(self.train_losses))]
        y = self.train_losses

        plt.plot(x, y)
        plt.savefig(path + '/losses/' + str(self.num_epoch) + '.png')

    def show_generaed(self, path):
        
        print("Generating image...")
        generated = self.model.forward(None)
        imgs = generated.clone().detach().to('cpu')
        grid_x = torchvision.utils.make_grid(imgs)
        cv2.imwrite(path + "/generated/" + str(self.num_epoch) + '.png', grid_x.permute(1, 2, 0).numpy() * 255)

    def epoch(self):

        for batch in tqdm(self.dataloader):

            if len(batch[1]) != self.batch_size:
                print("Skipping batch of size " + str(len(batch[1])))
                continue

            self.optimizer.zero_grad()

            image = batch[0]
            image = [torch.from_numpy(np.float32(i)) for i in image]
            image = torch.stack(image, dim = -1)
            image = image.to('cuda')
            idx = batch[1]

            code = self.get_codes(idx)

            self.model.set_code(code)

            reconstructed = torch.squeeze(self.model.forward(None))

            loss = torch.sum(torch.abs(image - reconstructed))

            loss.backward()

            self.optimizer.step()

            n_codes = self.model.get_code_content()

            for i in range(len(idx)):
                self.z_codes[idx[i]] = n_codes[i]

            self.train_losses.append(loss.detach().to('cpu').numpy())
