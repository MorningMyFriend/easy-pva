#!/usr/bin/env python
# -*- coding=utf-8 -*-

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import pprint
import  os
import aux_tools


def set_mode(mode = 'gpu', gpu_id = 0):
    aux_tools.set_mode(mode, gpu_id)

def change_test_prototxt(test_prototxt_file, cls_len):
    aux_tools.change_test_prototxt(test_prototxt_file, cls_len)

def test(classes, cfg_file, test_prototxt, test_model,
         comp_mode, max_per_image, vis,
         imdb_name = 'voc_2007_test',set_cfgs = None):

    if cfg_file is not None:
        cfg_from_file(cfg_file)
    if set_cfgs is not None:
        cfg_from_list(set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(test_model):
        print('{} is not existed...'.format(test_model))

    if test_prototxt is not None:
        aux_tools.change_test_prototxt(test_prototxt,len(classes)+1) # add 1 -- should consider background

    net = caffe.Net(test_prototxt, test_model, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(test_model))[0]

    imdb = get_imdb(imdb_name)
    imdb.competition_mode(comp_mode)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    test_net(net, imdb, max_per_image=max_per_image, vis=vis)