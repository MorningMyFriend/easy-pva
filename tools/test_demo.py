#!/usr/bin/env python
# -*- coding=utf-8 -*-

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import os.path as osp
from easy_module import aux_tools, test_net
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
                        action='store_true')  # 默认采用cpu模式
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--weights', dest='test_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--classes', dest='classes_name',
                        help='name of object classes', default=None)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    test_prototxt = ""
    test_model = ""
    cfg_file = ""
    CLASSES = []

    # set compute_mode
    if args.cpu_mode:
        test_net.set_mode('cpu')
    else:
        test_net.set_mode('gpu', args.gpu_id)
    # set solver prototxt
    if args.prototxt is not None:
        test_prototxt = args.prototxt
    #set model
    if args.test_model is not None:
        test_model = args.test_model
    #set cfg file
    if args.cfg_file is not None:
        cfg_file = args.cfg_file

    else:
        raise Exception('Please give the proto txt!')

    if args.classes_name is not None:
        CLASSES = aux_tools.get_classes(args.classes_name)
    else:
        raise  Exception('Please give the classes name txt!')

    test_net.test(CLASSES, cfg_file, test_prototxt, test_model,
                  args.comp_mode, args.max_per_image, args.vis, args.imdb_name, args.set_cfgs)
