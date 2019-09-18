#!/bin/bash
# add root_dir into PYTHONPATH
# 返回脚本文件放置的目录
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
# 将根目录添加之PYTHONPATH中
ROOT_PTAH=$SHELL_FOLDER/../
export PYTHONPATH=$PYTHONPATH:$ROOT_PATH

cd './data'
#
vocPath="VOCdevkit2007"
if [ -d "$vocPath" ]; then
# do nothing
echo "Had softlink"
else
echo "Create SymbolicLink VOCdevkit2007"
# should change the directory as yours
ln -s  "/home/cvrsg/VOCdataset/VOCdevkit" "$vocPath"
fi

# whether renew trainval/test
if [ -d "$vocPath/VOC2007/ImageSets/Main/test.txt" ]; then
# do nothing
echo "Had test.txt"
else
# make the label(general, test in a new dataset, so set trainval to 0, test 1)
python makeMain.py --trainval 0.9 --train 0.9
fi
# if results is not existed, make the directory
if [ -d "$vocPath/results/VOC2007/Main" ]; then
# do nothing
echo "Had folder 'results/VOC2007/Main'"
else
mkdir -p "$vocPath/results/VOC2007/Main"
fi

cd '..'
# test
python ./tools/test_demo.py --cpu --def model/voc2007_comp/test.pt --weights model/voc2007_comp/test.model --cfg model/voc2007_comp/test.yml --imdb voc_2007_test --classes data/VOCdevkit2007/classes_name.txt 2>&1 | tee log/log.txt
