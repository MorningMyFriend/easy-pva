#!/usr/bin/env python
# -*- coding=utf-8 -*-

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
import caffe
import argparse
import pprint
import numpy as np
import sys
import aux_tools

from caffe.proto import caffe_pb2
from google.protobuf import text_format

def combined_roidb(imdb_names, classes):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        # set classes of train in interface
        # no need into pascal_voc to reset self._classes
        imdb.set_classes(classes)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)

        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb

def set_mode(mode = 'gpu', gpu_id = 0):
    aux_tools.set_mode(mode, gpu_id)

def change_train_prototxt(train_prototxt_file, cls_len):
    aux_tools.change_train_prototxt(train_prototxt_file, cls_len)

def train(classes,
          cfg_file, solver_proto, train_prototxt, pretrained_model, max_iters, imdb_name = 'voc_2007_trainval',
          set_cfgs = None):
    if cfg_file is not None:
        cfg_from_file(cfg_file)
    if set_cfgs is not None:
        cfg_from_list(set_cfgs)
    if train_prototxt is not None:
        aux_tools.change_train_prototxt(train_prototxt,len(classes)+1) # add 1 -- should consider background
    imdb, roidb = combined_roidb(imdb_name, classes)
    print imdb.classes
    print '{:d} roidb entries'.format(len(roidb))

    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    train_net(solver_proto, roidb, output_dir,
              pretrained_model=pretrained_model,
              max_iters=max_iters)
