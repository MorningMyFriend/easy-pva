#!/bin/bash
# 返回脚本文件放置的目录
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
# 将根目录添加之PYTHONPATH中
ROOT_PTAH=$SHELL_FOLDER/../
export PYTHONPATH=$PYTHONPATH:$ROOT_PATH
# compress model 
# 对模型进行压缩
#python ./tools/compress_model.py --net model/voc2007/test.pt --weights model/voc2007/test.model --cfg model/voc2007/test.yml --classes model/voc2007/classes_name.txt 
python ./tools/compress_model.py --net model/Always/test.prototxt --weights model/Always/Always_iter_100000.caffemodel --cfg model/Always/test.yml --classes model/Always/classes_name.txt
