from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from model.supervisor import FCDNetSupervisor

parser = argparse.ArgumentParser()
# basic settings
parser.add_argument('--device',default='cuda:0',type=str)
parser.add_argument('--log_dir',default='data/model',type=str,help='')
parser.add_argument('--log_level',default='INFO',type=str)
parser.add_argument('--log_every',default=1,type=int)
parser.add_argument('--save_model',default=0,type=int)
#data settings
parser.add_argument('--batch_size',default=64,type=int)
parser.add_argument('--dataset_dir',default='data/solar_AL',type=str)
# model settings
parser.add_argument('--cl_decay_steps',default=2000,type=int)
parser.add_argument('--filter_type',default='dual_random_walk',type=str)
parser.add_argument('--horizon',default=12,type=int)
parser.add_argument('--seq_len',default=12,type=int)
parser.add_argument('--feas_dim',default=1,type=int)
parser.add_argument('--input_dim',default=1,type=int)
parser.add_argument('--ll_decay',default=0,type=int)
parser.add_argument('--max_diffusion_step',default=2,type=int)
parser.add_argument('--num_rnn_layers',default=1,type=int)
parser.add_argument('--output_dim',default=1,type=int)
parser.add_argument('--rnn_units',default=96,type=int)
parser.add_argument('--use_curriculum_learning',default=True,type=bool)
parser.add_argument('--embedding_size',default=256,type=int)
parser.add_argument('--kernel_size',default=5,type=int)
parser.add_argument('--freq',default=288,type=int)
parser.add_argument('--requires_graph',default=2,type=int)
parser.add_argument('--blocks',default=4,type=int)
parser.add_argument('--layers',default=2,type=int)
parser.add_argument('--level',default=4,type=int)
parser.add_argument('--dgraphs',default=10,type=float)
parser.add_argument('--graph_input_dim',default=1,type=int)
parser.add_argument('--graph_feas_dim',default=1,type=int)
parser.add_argument('--dataset',default='',type=str)
# train settings
parser.add_argument('--base_lr',default=0.003,type=float)
parser.add_argument('--dropout',default=0.3,type=float)
parser.add_argument('--epoch',default=0,type=int)
parser.add_argument('--epochs',default=250,type=int)
parser.add_argument('--epsilon',default=1.0e-3,type=float)
parser.add_argument('--global_step',default=0,type=int)
parser.add_argument('--lr_decay_ratio',default=0.1,type=float)
parser.add_argument('--max_grad_norm',default=5,type=int)
parser.add_argument('--max_to_keep',default=100,type=int)
parser.add_argument('--min_learning_rate',default=2.0e-05,type=float)
parser.add_argument('--optimizer',default='adam',type=str)
parser.add_argument('--patience',default=50,type=int)
parser.add_argument('--steps',default=[20, 30, 40],type=list)
parser.add_argument('--test_every_n_epochs', default=10, type=int)
parser.add_argument('--num_sample', default=10, type=int)
args = parser.parse_args()


if __name__ == '__main__':
    supervisor = FCDNetSupervisor(args=args)
    supervisor.train(args)
