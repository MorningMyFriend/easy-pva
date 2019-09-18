#!/usr/bin/env python
# -*- coding=utf-8 -*-
import os.path as osp
from easy_module import aux_tools, train_net
import argparse
import sys
from easydict import EasyDict as edict

this_dir = osp.dirname(__file__)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true') #默认采用cpu模式
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=100000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--classes', dest='classes_name',
                        help='name of object classes', default=None)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args




if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    solver_prototxt = ''
    pretrained_model = ''
    cfg_file = ''
    train_prototxt = ''
    CLASSES = []
    max_iters = 100000
    imdb_name = ''
    set_cfgs = ''

    # set compute_mode
    if args.cpu_mode:
        train_net.set_mode('cpu')
    else:
        train_net.set_mode('gpu', args.gpu_id)
    # set solver prototxt
    if args.solver is not None:
        solver_prototxt = args.solver
        """Load a config file and merge it into the default options."""
        import yaml
        with open(solver_prototxt, 'r') as f:
            solver_cfg = edict(yaml.load(f))
            train_prototxt = solver_cfg.train_net
    else:
        raise Exception('Please give the proto txt!')

    #set iteration nums
    max_iters = args.max_iters

    #set initial weights
    if args.pretrained_model is not None:
        pretrained_model = args.pretrained_model
    else:
        raise Exception('Please give the pretrained model weights')

    # set cfg_file
    if args.cfg_file is not None:
        cfg_file = args.cfg_file

    #set imdb name
    imdb_name =  args.imdb_name

    # set set_cfgs
    if args.set_cfgs is not None:
        set_cfgs= args.set_cfgs

    if args.classes_name is not None:
        CLASSES = aux_tools.get_classes(args.classes_name)
    else:
        raise  Exception('Please give the classes name txt!')

    # train the model
    train_net.train(classes = CLASSES,
                    cfg_file = cfg_file, solver_proto = solver_prototxt, train_prototxt = train_prototxt,
                    pretrained_model = pretrained_model, max_iters = max_iters, imdb_name = imdb_name, set_cfgs=set_cfgs)
