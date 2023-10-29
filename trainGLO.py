from trainer import GLO_Trainer, RandomCodeInitializer, Callback
from model import Generator
from data import IndexedDataset
from mnist_reader import MnistDataloader
from torch.utils.data import DataLoader
import argparse
import torch
import os
from torchviz import make_dot
import cv2
import numpy as np
import torchinfo

def options():

    parser = argparse.ArgumentParser(description='GLO')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    parser.add_argument('--code_dim', type=int, default=25, help='Code dimensionality')
    parser.add_argument('--filters', type=int, default=64, help='Number of filters')
    parser.add_argument('--out_chan', type=int, default=1, help='Number of output channels')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')

    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')

    return parser.parse_args()

if __name__ == '__main__':

    opt = options()

    os.makedirs(
        os.path.join(opt.results_dir, 'checkpoints'), exist_ok=True
    )

    os.makedirs(
        os.path.join(opt.results_dir, 'losses'), exist_ok=True
    )

    os.makedirs(
        os.path.join(opt.results_dir, 'generated'), exist_ok=True
    )

    dataset = MnistDataloader('mnist/train-images-idx3-ubyte/train-images-idx3-ubyte', 
                              'mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
                              'mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte', 
                              'mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    
    train, test = dataset.load_data()

    real_dataset = []
    for i in range(len(train[0])):
        if train[1][i] in [7,1]:
            real_dataset.append(train[0][i])

    real_dataset = real_dataset[:len(real_dataset) // 2]

    # cv2.imshow('img', np.uint8(real_dataset[0]))
    # cv2.waitKey(0)

    dataset = IndexedDataset(real_dataset)

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    model = Generator(opt.code_dim, opt.filters, opt.out_chan, opt.batch_size)

    torchinfo.summary(model, (opt.batch_size, opt.code_dim))
    input()

    # make_dot(model.forward()).view()
    # input()

    criterion = torch.nn.MSELoss(reduction='sum')

    initializer = RandomCodeInitializer(opt.seed, opt.code_dim)

    trainer = GLO_Trainer(model, dataset, dataloader, criterion, initializer)
    
    trainer.set_optimizer('adam', opt.lr, opt.beta1, opt.beta2)

    callbacks = [
        Callback(lambda trainer: trainer.save_checkpoint(opt.results_dir), 2),
        Callback(lambda trainer: trainer.save_losses(opt.results_dir), 5),
        Callback(lambda trainer: trainer.show_generaed(opt.results_dir), 5)
    ]

    trainer.train_for_epochs(opt.epochs, callbacks)



