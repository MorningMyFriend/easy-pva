#!/usr/bin/env python
# -*- coding=utf-8 -*-
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compress a Fast R-CNN network using truncated SVD."""

import _init_paths
import caffe
import numpy as np
import os, sys
import os.path as osp
from caffe.proto import caffe_pb2
import google.protobuf as pb
from google.protobuf import text_format
import cv2

def compress_weights(W, l):
    """Compress the weight matrix W of an inner product (fully connected) layer
    using truncated SVD.
    Parameters:
    W: N x M weights matrix
    l: number of singular values to retain
    Returns:
    Ul, L: matrices such that W \approx Ul*L
    """

    # numpy doesn't seem to have a fast truncated SVD algorithm...
    # this could be faster
    U, s, V = np.linalg.svd(W, full_matrices=False)

    Ul = U[:, :l]
    sl = s[:l]
    Vl = V[:l, :]

    L = np.dot(np.diag(sl), Vl)
    return Ul, L
#compress net
def compress_net(prototxt, prototxt_svd, caffemodel):
    # prototxt = 'models/VGG16/test.prototxt'
    # caffemodel = 'snapshots/vgg16_fast_rcnn_iter_40000.caffemodel'
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    # prototxt_svd = 'models/VGG16/svd/test_fc6_fc7.prototxt'
    # caffemodel = 'snapshots/vgg16_fast_rcnn_iter_40000.caffemodel'
    net_svd = caffe.Net(prototxt_svd, caffemodel, caffe.TEST)

    print('Uncompressed network {} : {}'.format(prototxt, caffemodel))
    print('Compressed network prototxt {}'.format(prototxt_svd))

    out = os.path.splitext(os.path.basename(caffemodel))[0] + '_svd'
    #out_dir = os.path.dirname(caffemodel)
    out_dir = os.path.dirname(prototxt_svd)

    # Compress fc6
    if net_svd.params.has_key('fc6_L'):
        l_fc6 = net_svd.params['fc6_L'][0].data.shape[0]
        print('  fc6_L bottleneck size: {}'.format(l_fc6))

        # uncompressed weights and biases
        W_fc6 = net.params['fc6'][0].data
        B_fc6 = net.params['fc6'][1].data

        print('  compressing fc6...')
        Ul_fc6, L_fc6 = compress_weights(W_fc6, l_fc6)

        # assert(len(net_svd.params['fc6_L']) == 1)

        # install compressed matrix factors (and original biases)
        net_svd.params['fc6_L'][0].data[...] = L_fc6

        net_svd.params['fc6_U'][0].data[...] = Ul_fc6
        net_svd.params['fc6_U'][1].data[...] = B_fc6

        #out += '_fc6_{}'.format(l_fc6)

    # Compress fc7
    if net_svd.params.has_key('fc7_L'):
        l_fc7 = net_svd.params['fc7_L'][0].data.shape[0]
        print '  fc7_L bottleneck size: {}'.format(l_fc7)

        W_fc7 = net.params['fc7'][0].data
        B_fc7 = net.params['fc7'][1].data

        print('  compressing fc7...')
        Ul_fc7, L_fc7 = compress_weights(W_fc7, l_fc7)

        # assert(len(net_svd.params['fc7_L']) == 1)

        net_svd.params['fc7_L'][0].data[...] = L_fc7

        net_svd.params['fc7_U'][0].data[...] = Ul_fc7
        net_svd.params['fc7_U'][1].data[...] = B_fc7

        #out += '_fc7_{}'.format(l_fc7)

    filename = '{}/{}.caffemodel'.format(out_dir, out)
    net_svd.save(filename)
    print 'Wrote svd model to: {:s}'.format(filename)

#generate merge model
def load_and_fill_biases(src_model, src_weights, dst_model, dst_weights):
    with open(src_model) as f:
        model = caffe.proto.caffe_pb2.NetParameter()
        pb.text_format.Merge(f.read(), model)

    for i, layer in enumerate(model.layer):
        if layer.type == 'Convolution': # or layer.type == 'Scale':
            # Add bias layer if needed
            if layer.convolution_param.bias_term == False:
                layer.convolution_param.bias_term = True
                layer.convolution_param.bias_filler.type = 'constant'
                layer.convolution_param.bias_filler.value = 0.0

    with open(dst_model, 'w') as f:
        f.write(pb.text_format.MessageToString(model))

    caffe.set_mode_cpu()
    net_src = caffe.Net(src_model, src_weights, caffe.TEST)
    net_dst = caffe.Net(dst_model, caffe.TEST)
    for key in net_src.params.keys():
        for i in range(len(net_src.params[key])):
            net_dst.params[key][i].data[:] = net_src.params[key][i].data[:]

    if dst_weights is not None:
        # Store params
        pass

    return net_dst

def merge_conv_and_bn(net, i_conv, i_bn, i_scale):
    # This is based on Kyeheyon's work
    assert(i_conv != None)
    assert(i_bn != None)

    def copy_double(data):
        return np.array(data, copy=True, dtype=np.double)

    key_conv = net._layer_names[i_conv]
    key_bn = net._layer_names[i_bn]
    key_scale = net._layer_names[i_scale] if i_scale else None

    # Copy
    bn_mean = copy_double(net.params[key_bn][0].data)
    bn_variance = copy_double(net.params[key_bn][1].data)
    num_bn_samples = copy_double(net.params[key_bn][2].data)

    # and Invalidate the BN layer
    net.params[key_bn][0].data[:] = 0
    net.params[key_bn][1].data[:] = 1
    net.params[key_bn][2].data[:] = 1
    if num_bn_samples[0] == 0:
        num_bn_samples[0] = 1

    if net.params.has_key(key_scale):
        print 'Combine {:s} + {:s} + {:s}'.format(key_conv, key_bn, key_scale)
        scale_weight = copy_double(net.params[key_scale][0].data)
        scale_bias = copy_double(net.params[key_scale][1].data)
        net.params[key_scale][0].data[:] = 1
        net.params[key_scale][1].data[:] = 0
    else:
        print 'Combine {:s} + {:s}'.format(key_conv, key_bn)
        scale_weight = 1
        scale_bias = 0

    weight = copy_double(net.params[key_conv][0].data)
    bias = copy_double(net.params[key_conv][1].data)
    alpha = scale_weight / np.sqrt(bn_variance / num_bn_samples[0] + np.finfo(np.double).eps)
    net.params[key_conv][1].data[:] = bias * alpha + (scale_bias - (bn_mean / num_bn_samples[0]) * alpha)
    for i in range(len(alpha)):
        net.params[key_conv][0].data[i] = weight[i] * alpha[i]

def merge_batchnorms_in_net(net):
    # for each BN
    for i, layer in enumerate(net.layers):
        if layer.type != 'BatchNorm':
            continue

        l_name = net._layer_names[i]

        l_bottom = net.bottom_names[l_name]
        assert(len(l_bottom) == 1)
        l_bottom = l_bottom[0]
        l_top = net.top_names[l_name]
        assert(len(l_top) == 1)
        l_top = l_top[0]

        can_be_absorbed = True

        # Search all (bottom) layers
        for j in xrange(i - 1, -1, -1):
            tops_of_j = net.top_names[net._layer_names[j]]
            if l_bottom in tops_of_j:
                if net.layers[j].type not in ['Convolution', 'InnerProduct']:
                    can_be_absorbed = False
                else:
                    # There must be only one layer
                    conv_ind = j
                    break

        if not can_be_absorbed:
            continue

        # find the following Scale
        scale_ind = None
        for j in xrange(i + 1, len(net.layers)):
            bottoms_of_j = net.bottom_names[net._layer_names[j]]
            if l_top in bottoms_of_j:
                if scale_ind:
                    # Followed by two or more layers
                    scale_ind = None
                    break

                if net.layers[j].type in ['Scale']:
                    scale_ind = j

                    top_of_j = net.top_names[net._layer_names[j]][0]
                    if top_of_j == bottoms_of_j[0]:
                        # On-the-fly => Can be merged
                        break

                else:
                    # Followed by a layer which is not 'Scale'
                    scale_ind = None
                    break


        merge_conv_and_bn(net, conv_ind, i, scale_ind)

    return net


def process_model(net, src_model, dst_model, func_loop, func_finally):
    with open(src_model) as f:
        model = caffe.proto.caffe_pb2.NetParameter()
        pb.text_format.Merge(f.read(), model)


    for i, layer in enumerate(model.layer):
        map(lambda x: x(layer, net, model, i), func_loop)

    map(lambda x: x(net, model), func_finally)

    with open(dst_model, 'w') as f:
        f.write(pb.text_format.MessageToString(model))


# Functions to remove (redundant) BN and Scale layers
to_delete_empty = []
def pick_empty_layers(layer, net, model, i):
    if layer.type not in ['BatchNorm', 'Scale']:
        return

    bottom = layer.bottom[0]
    top = layer.top[0]

    if (bottom != top):
        # Not supperted yet
        return

    if layer.type == 'BatchNorm':
        zero_mean = np.all(net.params[layer.name][0].data == 0)
        one_var = np.all(net.params[layer.name][1].data == 1)
        length_is_1 = (net.params['conv1_1/bn'][2].data == 1) or (net.params[layer.name][2].data == 0)

        if zero_mean and one_var and length_is_1:
            print 'Delete layer: {}'.format(layer.name)
            to_delete_empty.append(layer)

    if layer.type == 'Scale':
        no_scaling = np.all(net.params[layer.name][0].data == 1)
        zero_bias = np.all(net.params[layer.name][1].data == 0)

        if no_scaling and zero_bias:
            print 'Delete layer: {}'.format(layer.name)
            to_delete_empty.append(layer)

def remove_empty_layers(net, model):
    map(model.layer.remove, to_delete_empty)


# A function to add 'engine: CAFFE' param into 1x1 convolutions
def set_engine_caffe(layer, net, model, i):
    if layer.type == 'Convolution':
        if layer.convolution_param.kernel_size == 1\
            or (layer.convolution_param.kernel_h == layer.convolution_param.kernel_w == 1):
            layer.convolution_param.engine = dict(layer.convolution_param.Engine.items())['CAFFE']

def gen_merged_model(prototxt, caffemodel):
    output_prototxt = osp.splitext(prototxt)[0] + '_merge.prototxt'
    output_caffemodel = osp.splitext(caffemodel)[0] + '_merge.caffemodel'

    net = load_and_fill_biases(prototxt, caffemodel, prototxt + '.temp.pt', None)
    net = merge_batchnorms_in_net(net)
    process_model(net, prototxt + '.temp.pt', output_prototxt,
                  [pick_empty_layers, set_engine_caffe],
                  [remove_empty_layers])
    # Store params
    net.save(output_caffemodel)
    os.remove(prototxt + '.temp.pt')
    return output_prototxt,output_caffemodel

#change train prototxt
# change num_classes in train_prototxt by len of CLASSES (=> cls_len)
def change_train_prototxt(train_prototxt_file, cls_len):
    net = caffe_pb2.NetParameter()
    f = open(train_prototxt_file, 'r')
    text_format.Merge(f.read(), net)
    # judge whether num_classes is equal to len(CLASSES)
    num_str_left = ''
    for layer in net.layer:
        if layer.name == 'input-data':
            param_str = layer.python_param.param_str.encode('gbk')
            num_str =  filter(str.isdigit, param_str)
            num_cls = int(num_str)
            num_str_left = param_str[0:param_str.find(num_str)]
            if num_cls == cls_len:
                f.close()
                return True
            else:
                break
    # if not equal
    for layer in net.layer:
        if layer.name == 'input-data' or layer.name == 'roi-data':
            layer.python_param.param_str = num_str_left + str(cls_len)
        if layer.name == 'cls_score':
            layer.inner_product_param.num_output = cls_len
        if layer.name == 'bbox_pred':
            layer.inner_product_param.num_output = cls_len * 4

    f = open(train_prototxt_file, 'w')
    f.write(text_format.MessageToString(net))
    f.close()
    return False

#change test prototxt	
# change num_classes in test_prototxt by len of CLASSES (=> cls_len)
def change_test_prototxt(test_prototxt_file, cls_len):
    net = caffe_pb2.NetParameter()
    f = open(test_prototxt_file, 'r')
    text_format.Merge(f.read(), net)
    # judge whether num_classes is equal to len(CLASSES)
    num_str_left = ''
    for layer in net.layer:
        if layer.name == 'cls_score':
            num_cls = layer.inner_product_param.num_output
            if num_cls == cls_len:
                f.close()
                return True
            else:
                break
    # if not equal
    for layer in net.layer:
        if layer.name == 'cls_score':
            layer.inner_product_param.num_output = cls_len
        if layer.name == 'bbox_pred':
            layer.inner_product_param.num_output = cls_len * 4

    f = open(test_prototxt_file, 'w')
    f.write(text_format.MessageToString(net))
    f.close()
    return False

def take_picture(camera_id,width,height):
    capture = cv2.VideoCapture(camera_id)
    capture.set(cv2.CV_CAP_PROP_FRAME_WIDTH,width)
    capture.set(cv2.CV_CAP_PROP_FRAME_HEIGHT,height)
    success, frame = capture.read()
    while success == False:
        success, frame = capture.read()
    return frame

def get_classes(classes_path):
    f = open(classes_path)
    class_name = []
    while 1:
        name = f.readline()
        name = name.strip()
        class_name.append(name)
        if not name:
            break

    class_name.pop()
    #CLASSES = tuple(class_name)
    CLASSES = class_name
    f.close()
    return CLASSES

def set_mode(mode = 'gpu', gpu_id = 0):
    if mode == 'gpu':
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
    elif mode == 'cpu':
        caffe.set_mode_cpu()
    else:
        print "Please set mode = 'cpu' or 'gpu'"
