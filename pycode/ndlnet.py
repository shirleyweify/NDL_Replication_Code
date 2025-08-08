from __future__ import print_function
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Define a new activation function
# introduce non-linearity to the input tensor and enhance the representation power of the neural network
def new_gelu(x, jit=0.1):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    # x = (x0-jit).clone()
    # A1 = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    # x = (x0 + jit).clone()
    # A2 = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    # x = x0.clone()
    A = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    return A  # + A1 + A2


class eegXYPosAttn(nn.Module):
    def __init__(self):
        super().__init__()
        # Define learnable parameters
        # 4 random 2x2 tensors
        self.paramq0 = nn.Parameter(torch.randn((2, 2)), requires_grad=True)
        self.paramq1 = nn.Parameter(torch.randn((2, 2)), requires_grad=True)
        self.paramk0 = nn.Parameter(torch.randn((2, 2)), requires_grad=True)
        self.paramk1 = nn.Parameter(torch.randn((2, 2)), requires_grad=True)

    def forward(self, eegXYpos):
        elexy1 = eegXYpos @ self.paramq0  # matrix multiplication
        elexy1 = new_gelu(elexy1, 0.1)
        elecq = elexy1 @ self.paramq1
        elexy2 = eegXYpos @ self.paramk0
        elexy2 = new_gelu(elexy2, 0.1)
        eleck = elexy2 @ self.paramk1
        weight = F.softmax(elecq @ eleck.T / eleck.shape[1] ** (1 / 2), dim=1)  # function alpha
        return weight


# residual convolutional attention mechanism
# 3-layer 2d convolutional neural network
class rcAttn(nn.Module):
    def __init__(self, NC1, NC2, NC3, ksize1, ksize2):
        super().__init__()
        # nn.Conv2d: apply a 2d convolution over an input signal
        # in_channels: number of channels in the input image
        # out_channels: number of channels produced by convolution
        # kernel_size: size of the convolving kernel, (height, width) for the image
        # stride: strid of the convolution (the amount of movement over the image)
        self.uppCovq = nn.Sequential(nn.Conv2d(1, NC1, (ksize1, ksize2), (ksize1, ksize2)),
                                     nn.Conv2d(NC1, NC2, (ksize1, ksize2), (ksize1, ksize2)))
        self.lowCovq = nn.Sequential(nn.Conv2d(NC2, NC3, (ksize1, ksize2), (ksize1, ksize2)),
                                     nn.Conv2d(NC3, 1, (ksize1, ksize2), (ksize1, ksize2)))
        self.uppCovk = nn.Sequential(nn.Conv2d(1, NC1, (ksize1, ksize2), (ksize1, ksize2)),
                                     nn.Conv2d(NC1, NC2, (ksize1, ksize2), (ksize1, ksize2)))
        self.lowCovk = nn.Sequential(nn.Conv2d(NC2, NC3, (ksize1, ksize2), (ksize1, ksize2)),
                                     nn.Conv2d(NC3, 1, (ksize1, ksize2), (ksize1, ksize2)))
        self.uppCovv = nn.Sequential(nn.Conv2d(1, NC1, (ksize1, ksize2), (ksize1, ksize2)),
                                     nn.Conv2d(NC1, NC2, (ksize1, ksize2), (ksize1, ksize2)))
        self.lowCovv = nn.Sequential(nn.Conv2d(NC2, NC3, (ksize1, ksize2), (ksize1, ksize2)),
                                     nn.Conv2d(NC3, 1, (ksize1, ksize2), (ksize1, ksize2)))

    def forward(self, weightD):
        attNq = self.uppCovq(weightD)
        attNq = new_gelu(attNq, 0.1)
        attNq = self.lowCovq(attNq)

        attNk = self.uppCovk(weightD)
        attNk = new_gelu(attNk, 0.1)
        attNk = self.lowCovk(attNk)

        attNv = self.uppCovv(weightD)
        attNv = attNv + new_gelu(attNv, 0.1)
        attNv = self.lowCovv(attNv)
        attNv = attNv + new_gelu(attNv, 0.1)
        return attNq, attNk, attNv


# add a linear transformation
class rcAttnW(nn.Module):
    def __init__(self, NC1, NC2, NC3, ksize1, ksize2, n_rdim, n_embed):
        super().__init__()
        self.uppCovq = nn.Sequential(nn.Conv2d(1, NC1, (ksize1, ksize2), (ksize1, ksize2)),
                                     nn.Conv2d(NC1, NC2, (ksize1, ksize2), (ksize1, ksize2)))
        self.lowCovq = nn.Sequential(nn.Conv2d(NC2, NC3, (ksize1, ksize2), (ksize1, ksize2)),
                                     nn.Conv2d(NC3, 1, (ksize1, ksize2), (ksize1, ksize2)))
        self.Linear = nn.Linear(n_rdim, n_embed)
        # nn.Linear: in_features, size of input tensor; out_features, size of output tensor

    def forward(self, weightD):
        attNq = self.uppCovq(weightD)
        attNq = new_gelu(attNq, 0.1)
        attNq = self.lowCovq(attNq)
        attNq = new_gelu(attNq, 0.1)
        attNq = self.Linear(attNq)  # apply a linear transformation, learn weight for Y = XW + b
        # Y(nxo) = X(nxi)W(ixo) + b, n: batch size, i: num of input neurons, o: num of output neurons
        return attNq


# reshape and add a linear transformation
class rcAttnWR(nn.Module):
    def __init__(self, NC1, NC2, NC3, ksize1, ksize2, n_rdim, n_embed):
        super().__init__()
        self.uppCovq = nn.Sequential(nn.Conv2d(1, NC1, (ksize1, ksize2), (ksize1, ksize2)),
                                     nn.Conv2d(NC1, NC2, (ksize1, ksize2), (ksize1, ksize2)))
        self.lowCovq = nn.Sequential(nn.Conv2d(NC2, NC3, (ksize1, ksize2), (ksize1, ksize2)),
                                     nn.Conv2d(NC3, 1, (ksize1, ksize2), (ksize1, ksize2)))
        self.Linear = nn.Linear(n_rdim, n_embed)
        self.n_rdim = n_rdim

    def forward(self, weightD):
        attNq = self.uppCovq(weightD)
        attNq = new_gelu(attNq, 0.1) + attNq
        attNq = self.lowCovq(attNq)
        attNq = new_gelu(attNq, 0.1) + attNq
        attNq = attNq.reshape([-1, self.n_rdim])  # reshape tensor
        attNq = self.Linear(attNq)
        return attNq


# 3-layer linear neural network
class rcAttnW1(nn.Module):
    def __init__(self, n_rdim, n_embed):
        super().__init__()
        self.Linear1 = nn.Linear(n_rdim, n_rdim)
        self.Linear2 = nn.Linear(n_rdim, n_embed)

    def forward(self, weightD):
        attNq = self.Linear1(weightD)
        attNq = new_gelu(attNq, 0.1)
        attNq = self.Linear2(attNq)
        return attNq


class Net(nn.Module):
    def __init__(self, NC1A, NC2A, NC3A, NC1G, NC2G, NC3G, nX, nZ, ksizeX, ksizeZ, scale=1.0):  # nZ is half of the ncol of Z, e.g. 125
        super().__init__()
        # self.eegposlayer =eegXYPosAttn()
        n_rdim_X = math.floor(
            math.floor(math.floor(math.floor(nX / ksizeX) / ksizeX) / ksizeX) / ksizeX)  # num of dim X reduced to
        # self.AttnW = rcAttnW(NC1, NC2, NC3, 1, ksize_X, n_rdim_X, nZ * 2)  # 3-layer cnn, d x p
        self.AttnW = rcAttnW(NC1A, NC2A, NC3A, 1, ksizeX, n_rdim_X, nZ * 2)  # 3-layer cnn, d x p
        n_rdim_Z = math.floor(math.floor(
            math.floor(math.floor(nZ * 2 / ksizeZ) / ksizeZ) / ksizeZ) / ksizeZ)  # num of dim Z reduced to
        n_rdim_XZ = n_rdim_X * n_rdim_Z  # num of dim reduced to then vectorized
        # self.CNN = rcAttnWR(NC1 * 4, NC2 * 4, NC3 * 4, 2, 2, n_rdim_XZ, 1)  # 3-layer cnn, kernel size 1 for small p
        self.CNN = rcAttnWR(NC1G, NC2G, NC3G, ksizeZ, ksizeX, n_rdim_XZ, 1)
        self.dropoutR = nn.Dropout(0.0)  # for eval, reduce overfitting
        self.dropoutC = nn.Dropout(
            0.0)  # nn.Dropout(p=0.2): if applied on tensor x, simulate missing data, miss with prob = 0.2
        self.nX = nX
        self.nZ = nZ
        self.scale = scale

    def forward(self, data):
        midpoint = int(data.shape[-1] / 2)
        bgdata = torch.cat((data[:, :, :, int(midpoint - self.nX/2 - self.nZ):int(midpoint - self.nX/2)],
                            data[:, :, :, int(midpoint + self.nX/2):int(midpoint + self.nX/2 + self.nZ)]), dim=3)  # accelerator Z
        data = data[:, :, :, int(midpoint - self.nX/2):int(midpoint + self.nX/2)]  # X
        batchsize = data.shape[0]
        nc = data.shape[2]  # num of sensors
        nt = data.shape[3]  # num of time points
        # softmax: rescaling, make it range(0, 1), sum to 1
        weightM = self.AttnW(data).softmax(dim=2)  # get alpha(XW)
        # torch.transpose(m, n): transpose the mth and the nth dim
        # torch.sum(tensor, dim=(-2, -1)): sum the last two dim
        # torch.unsqueeze(m): add one dim on the mth dim
        # data = (weightM.transpose(-2, -1) @ data)
        data = (weightM.transpose(-2, -1) @ data) + torch.sum((weightM * bgdata), dim=(-2, -1)).unsqueeze(-1).unsqueeze(
            -1) * self.scale
        # torch.sigmoid: map the input to range(0, 1), output = 1 / (1 + exp(-input))
        output = torch.sigmoid(self.CNN(data))
        return output, weightM
