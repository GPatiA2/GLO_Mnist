import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, code_dim, filters, out_chan, batch_size):

        super(Generator, self).__init__()
        self.nf = filters
        self.code_dim = code_dim
        self.out_chan = out_chan

        self.z = torch.nn.parameter.Parameter(torch.randn(batch_size, self.code_dim, requires_grad=True).to('cuda'), requires_grad=True)

        self.model = nn.Sequential(
                
            self.get_dcn_block(self.code_dim, self.nf * 4, 4, 1, 0), # 4x4

            self.get_dcn_block(self.nf * 4, self.nf * 2, 4, 2, 1), # 8x8

            self.get_dcn_block(self.nf * 2, self.nf, 4, 2, 1), # 16x16

            nn.ConvTranspose2d(self.nf, self.out_chan, 4, 2, 1),

            nn.Tanh()
    
        )

    def forward(self, c):

        return self.model(self.z.view(self.z.shape[0], self.z.shape[1], 1, 1))
    
    def set_code(self, code):

        self.z.data = code

    def get_code_content(self):

        code = self.z.data
        code = torch.nn.functional.normalize(code, dim=0, p=2)
        return code

    def get_dcn_block(self, in_chan, out_chan, kernel_size, stride, padding):

        return nn.Sequential(

            nn.ConvTranspose2d(in_chan, out_chan, kernel_size, stride, padding),

            nn.BatchNorm2d(out_chan),

            nn.ReLU(True),

        )