import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, code_dim, filters, out_chan):

        super(Generator, self).__init__()
        self.nf = filters
        self.code_dim = code_dim
        self.out_chan = out_chan

        self.model = nn.Sequential(
                
            self.get_dcn_block(self.code_dim, self.nf * 8, 4, 1, 0), # 4x4

            self.get_dcn_block(self.nf * 8, self.nf * 4, 4, 2, 1), # 8x8

            self.get_dcn_block(self.nf * 4, self.nf * 2, 4, 2, 1), # 16x16

            self.get_dcn_block(self.nf * 2, self.nf, 4, 2, 1), # 32x32

            nn.ConvTranspose2d(self.nf, self.out_chan, 4, 2, 1),

            nn.Sigmoid()
    
        )

    def forward(self, code):

        return self.model(code.view(code.shape[0], code.shape[1], 1, 1))
        

    def get_dcn_block(self, in_chan, out_chan, kernel_size, stride, padding):

        return nn.Sequential(

            nn.ConvTranspose2d(in_chan, out_chan, kernel_size, stride, padding),

            nn.BatchNorm2d(out_chan),

            nn.ReLU(True),

        )