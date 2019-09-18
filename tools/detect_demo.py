# -*- coding:utf-8 -*-
#!/usr/bin/env python
import os
from easy_module import aux_tools, easy_detect
import cv2

import argparse
import sys

this_dir = os.path.dirname(__file__)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true') #默认采用cpu模式
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--net', dest='demo_net', help='Network to use',
                        default=None, type = str)
    parser.add_argument('--weights', dest='net_weights', help='Weights of Network',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--classes', dest='classes_name',
                        help='name of object classes', default=None)
    parser.add_argument('--pic_mode', dest='pic_mode',
                        help='input picture mode(single image, video or directory)\n'
                             'image - a single image\n'
                             'video - inpute a video\n'
                             'directory - a set of images',
                        choices=('image', 'video', 'directory'), default='image')
                        #choices=('image', 'video', 'dir', 'model_test'), default='image')
    parser.add_argument('--test_path', dest='test_path',
                        help='path of test image(s)', default=None)
    parser.add_argument('--result_dir', dest='result_dir',
                        help='dircectory of result', default=None)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def demo(image_path, wait_time):
    # just like the predefined data structure
    results = {}
    for cls_ind, cls in enumerate(CLASSES[1:]):
        results[cls] = 0

    image = cv2.imread(image_path)
    detections = easy_detect.detect(net, image, CLASSES, CONF_THRESH=0.7, NMS_THRESH=0.3)

    # display image with detections
    for detection in detections:
        cv2.rectangle(image, (int(detection[1]), int(detection[2])), (int(detection[3]), int(detection[4])),
                      (255, 255, 255), 2)
        cv2.putText(image, detection[0], (int(detection[1]), int(detection[2])), 0, 0.5, (0, 0, 255), 2)

    # output by the predefined data structure
    for detection in detections:
        results[detection[0]] = results[detection[0]] + 1
    print(results)

    base_name = os.path.basename(image_path)
    cv2.imshow(base_name, image)
    cv2.waitKey(int(wait_time))

    return image # for write a result

def demo_video(image, wait_time):
    # just like the predefined data structure
    results = {}
    for cls_ind, cls in enumerate(CLASSES[1:]):
        results[cls] = 0

    detections = easy_detect.detect(net, image, CLASSES, CONF_THRESH=0.7, NMS_THRESH=0.3)

    # display image with detections
    for detection in detections:
        cv2.rectangle(image, (int(detection[1]), int(detection[2])), (int(detection[3]), int(detection[4])),
                      (255, 255, 255), 2)
        cv2.putText(image, detection[0], (int(detection[1]), int(detection[2])), 0, 0.5, (0, 0, 255), 2)

    # output by the predefined data structure
    for detection in detections:
        results[detection[0]] = results[detection[0]] + 1
    print(results)

    cv2.imshow('image', image)
    cv2.waitKey(int(wait_time))

    return image # for write a result

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    cfg_file = ''
    net_pt = ''
    net_weights = ''
    save_path = ''
    CLASSES = ['__background__']
    test_path = ''
    result_dir = ''

    # set compute_mode
    if args.cpu_mode:
        easy_detect.set_mode('cpu')
    else:
        easy_detect.set_mode('gpu', args.gpu_id)

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

    # load model, just load once, or else it will be very slow
    easy_detect.change_test_prototxt(net_pt, len(CLASSES))
    net = easy_detect.load_net(cfg_file, net_pt, net_weights)

    # set cfg_file
    if args.test_path is not None:
        test_path = args.test_path
    else:
        raise Exception('Please give a test source(image, video or directory of images)!')

        # set cfg_file
    if args.result_dir is not None:
        result_dir = args.result_dir
    else:
        raise Exception('Please give a directory to get results!')


    if args.pic_mode == 'image':
        res = demo(test_path, 0)
        base_name = os.path.basename(test_path)
        cv2.imwrite(os.path.join(result_dir, base_name), res)

    elif args.pic_mode == 'video':
        cap = cv2.VideoCapture(test_path)

        # 获得码率及尺寸
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
        if fps <= 0 or fps > 60 :
            fps = 30
        base_name = os.path.basename(test_path)
        _name = os.path.splitext(base_name)[0] #无后缀
        # 指定写视频的格式, I420-avi, MJPG-mp4
        writer = cv2.VideoWriter(os.path.join(result_dir, _name+'.avi'),
                                 cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'),
                                 fps, size)
        while(True):
            ret, frame = cap.read()
            if ret == False:
                continue
            res = demo_video(frame, 1000.0/fps)
            writer.write(frame)
        cap.release()
        writer.release()

    elif args.pic_mode == 'directory': #dir
        image_names = os.listdir(test_path)
        for image_name in image_names:
            image_path = os.path.join(test_path, image_name)
            res = demo(image_path, 1000.0/30)
            cv2.imwrite(os.path.join(result_dir, image_name), res)
    #elif args.pic_mode == 'model_test': # test_path VOC2007的路径
    #    f = open(os.path.join(test_path, 'ImageSets', 'Main', 'test.txt'))
    #    image_names = []
    #    while 1:
    #        name = f.readline()
    #        name = name.strip()
    #        image_names.append(name+'.jpg')
    #        if not name:
    #            break
    #    image_names.pop()
    #    for image_name in image_names:
    #        image_path = os.path.join(test_path, 'JPEGImages', image_name)
    #        res = demo(image_path, 1000.0/30)
    #        cv2.imwrite(os.path.join(result_dir, image_name), res)
    



