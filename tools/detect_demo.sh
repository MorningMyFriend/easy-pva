#!/bin/bash
# add root_dir into PYTHONPATH
# 返回脚本文件放置的目录
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
# 将根目录添加之PYTHONPATH中
ROOT_PTAH=$SHELL_FOLDER/../
export PYTHONPATH=$PYTHONPATH:$ROOT_PATH

# single image
#python ./tools/detect_demo.py --gpu 0 --net model/voc2007_comp/test.pt --weights model/voc2007_comp/test.model --cfg model/voc2007_comp/test.yml --classes model/voc2007_comp/classes_name.txt --pic_mode image --test_path data/demo/000456.jpg --result_dir data/result 2>&1 | tee log/log.txt 

## video
#python ./tools/detect_demo.py --gpu 0 --net model/voc2007_comp/test.pt --weights model/voc2007_comp/test.model --cfg model/voc2007_comp/test.yml --classes model/voc2007_comp/classes_name.txt --pic_mode video --test_path data/Cross.mp4 --result_dir data/result 2>&1 | tee log/log.txt

## directory
#python ./tools/detect_demo.py --gpu 0 --net model/voc2007_comp/test.pt --weights model/voc2007_comp/test.model --cfg model/voc2007_comp/test.yml --classes model/voc2007_comp/classes_name.txt --pic_mode directory --test_path data/demo --result_dir data/result 2>&1 | tee log/log.txt 

## model_test
#python ./tools/detect_demo.py --gpu 0 --net model/voc2007_comp/test.pt --weights model/voc2007_comp/test.model --cfg model/voc2007_comp/test.yml --classes model/voc2007_comp/classes_name.txt --pic_mode model_test --test_path data/VOCdevkit2007/VOC2007 --result_dir data/result 2>&1 | tee log/log.txt   
python ./tools/detect_demo.py --gpu 0 --net model/voc2007_comp/test.prototxt --weights model/Always/Always_iter_100000.caffemodel --cfg model/voc2007_comp/test.yml --classes data/VOCdevkit2007/VOC2007/classes_name.txt --pic_mode directory --test_path data/VOCdevkit2007/VOC2007 --result_dir data/result 2>&1 | tee log/log.txt   
