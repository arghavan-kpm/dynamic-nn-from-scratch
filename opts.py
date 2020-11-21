import argparse
import os
import sys

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--num_epochs', type=int, default=2,
                                 help='total training epochs.')
        self.parser.add_argument('--optimizer', default='GD',
                                 help='training optimizer.', choices=['SGD', 'GD'])
        self.parser.add_argument('--layers', default='784, 100, 10',
                                 help='number of neurons in each layer, '
                                      'use comma between them.')
        self.parser.add_argument('--active_funcs', default='linear,sigmoid',
                                 help='activation function for each layer, '
                                      'use comma between them.')
        self.parser.add_argument('--regularizers', default='dropOut,nothing',
                                 help='regularizers for each layer, '
                                      'use comma between them')
        self.parser.add_argument('--lr', type=float, default=0.7,
                             help='learning rate.')
        self.parser.add_argument('--dataset_dir', default='./data',
                             help='path to dataset.')

    def parse(self):
        opt = self.parser.parse_args()
        opt.layers = [int(l) for l in opt.layers.split(',')]
        opt.active_funcs = [af for af in opt.active_funcs.split(',')]
        opt.regularizers = [reg for reg in opt.regularizers.split(',')]

        print(
            f"############################################################################",
            f"#",
            f"#  A neural network with {len(opt.layers)} layers is created.",
            sep='\n'
            )
        for i in range(len(opt.layers)):
            if i == 0:
                print(f"#  layer({i} - input image): {opt.layers[i]} neurons.")
            else:
                print(f"#  layer({i}): {opt.layers[i]} neurons, {opt.active_funcs[i-1]} activation function, ",
                      f"with {opt.regularizers[i-1][:opt.regularizers[i-1].find('n')+2] if 'n' in opt.regularizers[i-1] else opt.regularizers[i-1]}",
                      f" regularizer.")

        GD = "Batch Gradient Descent"
        SGD = "Stochastic Gradient Descent"
     
        print(
            f"#",
            f"#  learning rate is {opt.lr}.",
            f"#  dataset is at {opt.dataset_dir}.",
            f"#  optimizer is {SGD if 'S' in opt.optimizer else GD}",
            f"#",
            sep='\n'
            )

        return opt

