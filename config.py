import argparse
import torch
from datetime import datetime
import re

class Config():
    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser(description='Argparse single GCN')
        timestamp = datetime.today().strftime("%Y%m%d%H%M%S")

        # pytorch base
        parser.add_argument("--cuda_num", default=2, type=str, help="0~5")
        parser.add_argument("--device", default='cuda', type=str, help="'cuda' or 'cpu'")

        # training hyperparams
        parser.add_argument('--seed', type=int, default=88, help='random seed')
        parser.add_argument('--fold_num', '-fold_num', type=int, default=5, help='the fold_num')
        parser.add_argument('--load', '-load', type=str, default='best_weights.pth', help='Load model from a .pth file')
        parser.add_argument("--dropout_ratio", default=0.2, type=float, help="fc layer dropout")

        parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
        parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
        parser.add_argument("--batch_size", default=32, type=int, help="batch size")
        parser.add_argument('--num_epoch', default= 250, type=int, help='num_epoch')
        parser.add_argument('--epoch_check', default=1, type=int, help='step of test function')

        parser.add_argument('--atlas', default="AAL", type=str, help='atlas type')
        parser.add_argument('--numROI', default=116, type=str, help='atlas type')

        parser.add_argument('--multi_atlas', default=["AAL","Craddock"], type=str, help='atlas type')
        parser.add_argument('--multi_numROI', default=[116,200], type=str, help='atlas type')
        parser.add_argument('--keep_nodes', default=[[23,24,35,36,65,67,68],[22,26,30,31,35,36],[80,127,147,156,167]], type=str, help='atlas main roi')
        parser.add_argument('--multi_keep', default=[[],[]], type=str, help='atlas main roi')
        parser.add_argument('--sub_graph', default=6, type=str, help='sub_graph num')
         


        self.args = parser.parse_args()
        self.cuda_num = self.args.cuda_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed = self.args.seed
        self.fold_num = self.args.fold_num
        self.load = self.args.load
        self.dropout_ratio = self.args.dropout_ratio

        self.lr = self.args.lr
        self.weight_decay = self.args.weight_decay
        self.batch_size = self.args.batch_size
        self.num_epoch = self.args.num_epoch
        self.epoch_check = self.args.epoch_check
        # self.sampling = self.args.sampling

        self.atlas = self.args.atlas
        self.numROI = self.args.numROI

        self.multi_atlas = self.args.multi_atlas
        self.multi_numROI = self.args.multi_numROI
        self.keep_nodes = self.args.keep_nodes
        self.multi_keep = self.args.multi_keep
        self.sub_graph = self.args.sub_graph