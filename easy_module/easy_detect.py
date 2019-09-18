#!/usr/bin/env python
# -*- coding=utf-8 -*-
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""
import _init_paths
import aux_tools
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

from caffe.proto import caffe_pb2
from google.protobuf import text_format

def demo(net, im, _t, CLASSES, CONF_THRESH = 0.7, NMS_THRESH = 0.3):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, _t)
    timer.toc()
    
    detectrions_result = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        detections = get_detections(cls, dets, thresh=CONF_THRESH)
        #detections = get_detections(cls_ind, dets, thresh=CONF_THRESH)
        if len(detections) == 0:
            continue
        else:
            detectrions_result.extend(detections)
    return detectrions_result

def get_detections(class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return []
    detections = []
    for i in inds:
        bbox1 = dets[i, :4]
        bbox = bbox1.tolist()
        score1 = dets[i, -1]
        score = score1.tolist()
        detection = [class_name, bbox[0], bbox[1], bbox[2], bbox[3], score]
        detections.append(detection)
        
    return detections
#模式设置代码复用
def set_mode(mode = 'gpu', gpu_id = 0):
    aux_tools.set_mode(mode, gpu_id)
#prototxt修改代码复用
def change_test_prototxt(test_prototxt_file, cls_len):
    aux_tools.change_test_prototxt(test_prototxt_file, cls_len)

def load_net(cfg_file, net_pt, net_weight):
    cfg_from_file(cfg_file)
    net = caffe.Net(net_pt, net_weight, caffe.TEST)
    return net

def detect(net, img, CLASSES, CONF_THRESH = 0.7, NMS_THRESH = 0.3):
    _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
    detections = demo(net, img, _t, CLASSES, CONF_THRESH, NMS_THRESH)
    return detections
