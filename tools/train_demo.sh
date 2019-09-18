#!/bin/bash
# add root_dir into PYTHONPATH
# 返回脚本文件放置的目录
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
# 将根目录添加之PYTHONPATH中
ROOT_PTAH=$SHELL_FOLDER/../
export PYTHONPATH=$PYTHONPATH:$ROOT_PATH
# delete data cache
cd './data'
cache='./cache'
if [ -d "$cache" ]; then
echo "Delete the cache"
rm -r "$cache"
fi
#
vocPath="VOCdevkit2007"
if [ -d "$vocPath" ]; then  
echo "Delete the symbolicLink"
rm -r "$vocPath"  
fi 
echo "Create SymbolicLink VOCdevkit2007"
# should change the directory as yours
ln -s  "/home/jphu/TestGround2018/easy-pvanet/data/voc2007" "$vocPath"
cd '..'
# make the label
python ./data/makeMain.py --trainval 0.9 --train 0.8

# train
python ./tools/train_demo.py --gpu 0 --solver model/train/solver.prototxt --iters 100000 --weights model/train/original_train.model --cfg model/train/train.yml --imdb voc_2007_trainval --classes data/VOCdevkit2007/classes_name.txt 2>&1 | tee log/log.txt 
