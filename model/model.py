import torch
import torch.nn as nn
from torch.nn import functional as F
from model.cell import DCGRUCell, SmoothSparseUnit
import numpy as np
import math
import pywt
from .STFE import *
from .LTFE import *

class Seq2SeqAttrs:
    def __init__(self, args):
        #self.adj_mx = adj_mx
        self.max_diffusion_step = args.max_diffusion_step
        self.cl_decay_steps = args.cl_decay_steps
        self.filter_type = args.filter_type
        self.num_nodes = args.num_nodes
        self.num_rnn_layers = args.num_rnn_layers
        self.rnn_units = args.rnn_units
        self.output_dim = args.output_dim
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, args):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, args)
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.seq_len = args.seq_len  # for the encoder
        self.device = args.device
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type, device=self.device) for _ in range(self.num_rnn_layers)])


    def forward(self, inputs, adj, hidden_state=None):
        """
        Encoder forward pass.
        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, args):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, args)
        self.output_dim = args.output_dim
        self.horizon = args.horizon  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.device = args.device
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type, device=self.device) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj, hidden_state=None):
        """
        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class FISMFModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, node_feas, logger, args):
        super().__init__()
        Seq2SeqAttrs.__init__(self, args)
        self.args = args
        self.encoder_model = EncoderModel(args)
        self.decoder_model = DecoderModel(args)
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = args.use_curriculum_learning
        self._logger = logger
        self.embedding_size = args.embedding_size
        self.seq_len = args.seq_len
        self.feas_dim = args.feas_dim
        self.input_dim = args.input_dim
        self.kernel_size = args.kernel_size
        self.batch_size= args.batch_size
        self.freq = args.freq
        self.requires_graph = args.requires_graph
        self.level = args.level
        self.num_nodes = args.num_nodes
        self.device = args.device
        self.graphs = self.requires_graph
        self.instant_forecasting = Instant_forecasting(args)
        self.output_dim = args.output_dim
        self.dataset = args.dataset
        self.epis = []
        for index in range(self.requires_graph):
            self.epis.append(nn.Parameter(torch.tensor([0.30], device=self.device), requires_grad=True))
        self.ShortAdj_generator = ShortAdj_generator(args, node_feas)
        self.LongAdj_generator = LongAdj_generator(args, node_feas)
        self.lambdax = torch.nn.Parameter(torch.tensor([0.30], device=self.device), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.tensor([0.10]), requires_grad=True).to(self.device)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, adj):
        """
        Encoder forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.args.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], adj, encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, adj, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol
        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, adj, decoder_hidden_state)

            decoder_input = decoder_output

            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, labels=None, batches_seen=None, epoch=None):
        """
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """

        ShortAdj = self.ShortAdj_generator(inputs)
        Long_Adj = self.LongAdj_generator(inputs)
        adj_list = (ShortAdj + self.lambdax * Long_Adj)/ (1.0+self.lambdax)
        adj = adj_list[0]
        for index in range(1, self.requires_graph):
            adj = adj + self.epis[index] * adj_list[index] # 这里可以设计一个更为复杂的fusion模块
        encoder_hidden_state = self.encoder(inputs, adj)
        outputs = self.decoder(encoder_hidden_state, adj, labels, batches_seen=batches_seen)
        res, adj = self.instant_forecasting(inputs)
        res = res.squeeze(dim=3).permute(1, 0, 2)
        outputs = (outputs + self.beta*res)/ (1.0+self.beta)
        return outputs, adj