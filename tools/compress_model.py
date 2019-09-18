#!/usr/bin/env python
# -*- coding:utf-8 -*-
from easy_module import aux_tools
import os.path as osp
import os
import shutil
import argparse
import sys

this_dir = osp.dirname(__file__)

net_pt_svd = osp.join(this_dir, '..', 'model','comp_model', 'merge_svd.prototxt')

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Compress a Fast R-CNN network')
    parser.add_argument('--net', dest='demo_net', help='Network to use',
                        default=None, type = str)
    parser.add_argument('--weights', dest='net_weights', help='Weights of Network',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--classes', dest='classes_name',
                        help='name of object classes', default=None)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__== '__main__':
    args = parse_args()

    CLASSES = ['__background__']

    net_pt = ''
    net_weights = ''
    cfg_file = ''
    classes_path = ''

    if args.demo_net is not None:
        net_pt = args.demo_net
    else:
        raise  Exception('Please give the network file!')

    if args.net_weights is not None:
        net_weights = args.net_weights
    else:
        raise  Exception('Please give the weight file!')

    # set cfg_file
    if args.cfg_file is not None:
        cfg_file = args.cfg_file
    else:
        raise Exception('Please give the configure file!')

    if args.classes_name is not None:
        CLASSES.extend(aux_tools.get_classes(args.classes_name))
    else:
        raise  Exception('Please give the classes name txt!')
    
    aux_tools.change_test_prototxt(net_pt,len(CLASSES))
    aux_tools.change_test_prototxt(net_pt_svd, len(CLASSES))
    output_pt,output_weight = aux_tools.gen_merged_model(net_pt, net_weights)
    aux_tools.compress_net(output_pt,net_pt_svd,output_weight)
    #remove template files
    os.remove(output_pt)
    os.remove(output_weight)
    #copy cfg file
    # output directory
    out_dir = os.path.dirname(net_pt_svd)
    #configure file basename
    cfg_name = os.path.basename(cfg_file)
    #to check whether a .yml file in output directoty
    file_list = os.listdir(out_dir)
    exist_flag = False
    for file in file_list:
        if file.endswith('.yml'):
            flag = True
    #copy .yml
    if osp.exists(cfg_file) and exist_flag == False:
        out_cfg = osp.join(out_dir, cfg_name)
        shutil.copy(cfg_file,out_cfg)
