import torch
import torch.nn as nn
from torch.nn import functional as F
from model.cell import SmoothSparseUnit
import numpy as np
import math
import pywt
from .ST_Norm import *
import math
minm = -99999999

def WT_his(device, level, Stats):
    array = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.05, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.05]
    ])
    Stats = Stats.detach().cpu().numpy()
    if len(Stats.shape) == 2:
        Stats = np.expand_dims(Stats, axis=2)
    time_steps, num_nodes, feas_dim = Stats.shape
    result = []
    for feas in range(feas_dim):
        Data = Stats[:, :, feas]
        info = []
        for i in range(num_nodes):
            Wavelet_res = []
            SingleSampleDataWavelet = Data[:, i]
            coeffs = pywt.wavedec(SingleSampleDataWavelet, 'db{}'.format(level), level=level)
            for index in range(level + 1):
                factors = []
                for j in range(len(coeffs)):
                    factors.append(coeffs[j] * array[index][j])
                y = pywt.waverec(factors, 'db{}'.format(level))
                Wavelet_res.append(torch.Tensor(y))
            res = torch.stack(Wavelet_res, dim=1).to(device)
            info.append(res)
        result.append(torch.stack(info, dim=1))
    results = torch.stack(result, dim=3)
    return results

class ShortAdj_generator(nn.Module):
    def __init__(self, args, time_series, reduction_ratio=16, alpha=0.2, dropout_rate=0.5):
        super(ShortAdj_generator, self).__init__()
        self.freq = args.freq
        self.kernel_size = args.kernel_size
        self.num_nodes = args.num_nodes
        self.embedding = args.embedding_size
        self.time_series = time_series
        self.seq_len = args.seq_len
        self.feas_dim = args.graph_feas_dim
        self.input_dim = args.graph_input_dim
        self.segm = int(self.time_series.shape[0] // self.freq)
        self.graphs = args.requires_graph
        self.device = args.device
        self.level = args.level
        self.expand_dim = self.level + 1
        self.delta_series = torch.zeros_like(self.time_series).to(self.device)
        self.conv1d = nn.Conv1d(in_channels=self.expand_dim * self.segm * self.feas_dim, out_channels=self.graphs, kernel_size=self.kernel_size, padding=0)
        self.fc_1 = nn.Linear(self.freq - self.kernel_size + 1, self.embedding)
        self.fc_2 = nn.Linear(self.embedding, self.embedding // reduction_ratio)
        self.fc_3 = nn.Linear(self.embedding // reduction_ratio, self.num_nodes)
        self.snorm = SNorm(self.freq)
        self.tnorm = TNorm(self.num_nodes, self.freq)
        self.pre_process()

    def pre_process(self):
        for i in range(self.time_series.shape[0]):
            if i == 0:
                self.delta_series[i] = self.time_series[i]
            else:
                self.delta_series[i] = self.time_series[i]-self.time_series[i-1]
        self.wave_list = []
        for i in range(self.segm):
            time_seg = self.delta_series[i * self.freq + 1: (i + 1) * self.freq + 1] # [self.freq, self.num_nodes, self.input_dim]
            feas = WT_his(self.device, self.level, time_seg)
            self.wave_list.append(feas)
        self.His_data = torch.stack(self.wave_list, dim=0) # [segm, freq, num_nodes, expand_dim, feas_dim]


    def forward(self, node_feas): # input: (seq_len, batch_size, num_sensor * input_dim)
        t = self.His_data.reshape(self.segm, self.freq, self.num_nodes, -1)
        t = self.snorm(t)
        t = self.tnorm(t)
        self.times = t.permute(2, 0, 3, 1).reshape(self.num_nodes, -1, self.freq)
        mid_input =  self.conv1d(self.times).permute(1, 0, 2) # (graphs, num_nodes, freq-kernel_size+1)
        mid_output = torch.stack([F.relu(self.fc_1(mid_input[i,...])) for i in range(self.graphs)], dim=0)
        mid_output = torch.sigmoid(self.fc_2(mid_output))
        output = SmoothSparseUnit(self.fc_3(mid_output), 1, 0.02)
        return output

class LongAdj_generator(nn.Module):
    def __init__(self, args, time_series, reduction_ratio=16, alpha=0.2, dropout_rate=0.5):
        super(LongAdj_generator, self).__init__()
        self.freq = args.freq
        self.kernel_size = args.kernel_size
        self.num_nodes = args.num_nodes
        self.embedding = args.embedding_size
        self.time_series = time_series
        self.seq_len = args.seq_len
        self.feas_dim = args.graph_feas_dim
        self.input_dim = args.graph_input_dim
        self.segm = int(self.time_series.shape[0] // self.freq)
        self.graphs = args.requires_graph
        self.device = args.device
        self.level = args.level
        self.expand_dim = self.level + 1
        self.delta_series = torch.zeros_like(self.time_series).to(self.device)
        self.conv1d = nn.Conv1d(in_channels=self.expand_dim * self.freq * self.feas_dim, out_channels=self.graphs, kernel_size=self.kernel_size, padding=0)
        self.fc_1 = nn.Linear(self.segm - self.kernel_size + 1, self.embedding)
        self.fc_2 = nn.Linear(self.embedding, self.embedding // reduction_ratio)
        self.fc_3 = nn.Linear(self.embedding // reduction_ratio, self.num_nodes)
        self.snorm = SNorm(self.segm)
        self.tnorm = TNorm(self.num_nodes, self.segm)
        self.pre_process()

    def pre_process(self):
        for i in range(self.time_series.shape[0]):
            if i == 0:
                self.delta_series[i] = self.time_series[i]
            else:
                self.delta_series[i] = self.time_series[i] - self.time_series[i - 1]
        self.wave_list = []
        for i in range(self.segm):
            time_seg = self.delta_series[i * self.freq + 1: (i + 1) * self.freq + 1]  # [self.freq, self.num_nodes, self.input_dim]
            feas = WT_his(self.device, self.level, time_seg)
            self.wave_list.append(feas)
        self.His_data = torch.stack(self.wave_list, dim=0)  # [segm, freq, num_nodes, expand_dim, feas_dim]

    def forward(self, node_feas): # input: (seq_len, batch_size, num_sensor * input_dim)
        t = self.His_data.reshape(self.segm, self.freq, self.num_nodes, -1).permute(1, 0, 2, 3)
        t = self.snorm(t)
        t = self.tnorm(t)
        self.times = t.permute(2, 0, 3, 1).reshape(self.num_nodes, -1, self.segm)
        mid_input =  self.conv1d(self.times).permute(1, 0, 2) # (graphs, num_nodes, freq-kernel_size+1)
        mid_output = torch.stack([F.relu(self.fc_1(mid_input[i,...])) for i in range(self.graphs)], dim=0)
        mid_output = torch.sigmoid(self.fc_2(mid_output))
        output = SmoothSparseUnit(self.fc_3(mid_output), 1, 0.02)
        return output



