import torch
import torch.nn as nn
from torch.nn import functional as F
from model.cell import DCGRUCell, SmoothSparseUnit
import numpy as np
import torch.fft
import math
import pywt
from .ST_Norm import *

class nconv(nn.Module):
    def __init__(self):
       super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class FAGCN(nn.Module):
    def __init__(self,args,c_in,c_out,dropout=0.3,order=2, eps=0.3):
        super(FAGCN,self).__init__()
        self.nconv = nconv()
        c_in = (order*2)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order
        self.eps = eps
        self.eye = torch.eye(args.num_nodes).to(args.device)

    def forward(self,x,adj):
        out = []
        L = self.eps * self.eye + adj
        H = self.eps * self.eye - adj
        support = [L,H]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class SpectralConv1d(nn.Module):
    def __init__(self, args):
        super(SpectralConv1d, self).__init__()
        self.num_nodes = args.num_nodes
        self.seq_len = args.seq_len
        self.in_channels = args.batch_size * args.input_dim
        self.out_channels = args.dgraphs
        self.batch_size = args.batch_size
        self.Lin = nn.Linear(self.in_channels, self.out_channels)

    def forward(self, x):
        # [num_nodes, seq_len, input_dim*batch_size]
        ffted = torch.view_as_real(torch.fft.fft(x, dim=1))
        # ffted.shape: (batch, seq_len, input_dim*batch_size)
        real = self.Lin(ffted[..., 0].contiguous())
        img = self.Lin(ffted[..., 1].contiguous())
        time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)
        iffted = torch.fft.irfft(torch.view_as_complex(time_step_as_inner), n=time_step_as_inner.shape[1], dim=1)
        Am = torch.sqrt(real*real+img*img)
        S = torch.atan(real/(img+0.0001))
        return iffted,Am,S


class Instant_graph(nn.Module):
    def __init__(self, args):
        super(Instant_graph, self).__init__()
        self.num_nodes = args.num_nodes
        self.seq_len = args.seq_len
        self.output_channel = args.input_dim
        self.time_step = args.seq_len
        self.dgraphs = args.dgraphs
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.device = args.device
        self.level = args.level
        self.requires_graph = args.requires_graph
        self.kernel_size = args.kernel_size
        self.spectconv = SpectralConv1d(args)
        self.Linear = nn.Linear(self.seq_len, 1)
        self.fc1 = nn.Linear(self.dgraphs, self.num_nodes)
        self.fc2 = nn.Linear(self.dgraphs, self.num_nodes)
        self.fc3 = nn.Linear(self.dgraphs, self.num_nodes)


    def forward(self, x):
        # x.shape: [num_nodes, batch_size*input_dim, seq_len]
        out, Am, S = self.spectconv(x)
        mid_input = SmoothSparseUnit((self.fc1(out)+self.fc2(Am)+self.fc3(S)),1, 0.02)
        mid_input = mid_input.permute(0, 2, 1)
        graph = SmoothSparseUnit((torch.squeeze(self.Linear(mid_input))),1,0.02)
        return graph


class Instant_forecasting(nn.Module):
    def __init__(self, args, channels=16, kernel_size=2):
        super(Instant_forecasting, self).__init__()
        self.dropout = args.dropout
        self.blocks = args.blocks
        self.layers = args.layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.num_nodes = args.num_nodes
        self.seq_len = args.seq_len
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.out_dim = args.horizon
        self.horizon = args.horizon
        self.gconv = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.device = args.device
        self.start_conv = nn.Conv2d(in_channels=self.input_dim,
                                    out_channels=channels,
                                    kernel_size=(1, 1))
        self.graphs = args.requires_graph
        self.instant_graph = Instant_graph(args)
        receptive_field = 1

        self.supports_len = 0

        for b in range(self.blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=channels,
                                                   out_channels=channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=channels,
                                                 out_channels=channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))


                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=channels,
                                                 out_channels=channels,
                                                 kernel_size=(1, 1)))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.bn.append(nn.BatchNorm2d(channels))
                self.gconv.append(FAGCN(args,channels, channels))

        self.end_conv_1 = nn.Conv2d(in_channels=channels,
                                    out_channels=channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=channels,
                                    out_channels=self.out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):
        # the required input shape: [batch_size, self.input_dim, self.num_nodes, self.seq_len]
        input = input.reshape(self.seq_len, self.batch_size, self.num_nodes, -1)
        adj = self.instant_graph(input.permute(2,3,1,0).reshape(self.num_nodes, self.seq_len, -1))
        input = input.permute(1, 3, 2, 0)

        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            residual = x
            # dilated convolution
            # x.shape:[batch_size, residual_channels, num_nodes, seq_len]
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            x = self.gconv[i](x, adj)
            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        # x.shape: [batch_size, output_dim, num_nodes, 1]
        return x, adj