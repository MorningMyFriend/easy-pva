# easy-pvanet

## 描述

基于Pvanet目标检测代码进行修改，使得PVANET能够一键训练和一键测试。



## 编译

需要对`caffe-fast-rcnn`和`lib`进行编译。

* `caffe-fast-rcnn`编译

  * 根据官方网站http://caffe.berkeleyvision.org/installation.html进行`caffe-fast-rcnn`配置编译

  * 编译的基本步骤：

    ```shell
    make -j 
    make pycaffe 
    make distribute
    ```

  * 编译后的文件保存在`easy-pvanet/caffe-fast-rcnn/distribute`文件夹下

* `lib`编译

  * `make`



## 基本功能点

### 1. 模型训练

#### 命令

```shell
cd easy-pvanet
./tools/train_demo.sh
```

#### 参数修改

* 修改训练数据集的软链接`ln -s  "/home/cvrsg/VOCdataset/Hand/hand_dataset" "$vocPath"`，将`/home/cvrsg/VOCdataset/Hand/hand_dataset`设置为自己的数据集；

* 训练数据集中需要包含的文件结构

  ```shell
  hand_dataset
  |
  |____ VOC2007
  |     |
  |     |____ Annotations
  |     |
  |     |____ ImageSets
  |     |     |
  |     |     |____ Main
  |     |
  |     |____ JPEGImages
  |
  |____ classes_name.txt
  ```

#### 预训练模型下载

* 下载地址：萃舟服务器中`model-zoo/easy-pvanet/model/train/original_train.model`
* 将预训练模型放置在`easy-pvanet/model/train`文件夹下，并命名为`original_train.model`。

### 2. 目标检测测试

#### 命令

```shell
cd easy-pvanet
./tools/detect_demo.sh
```

#### 测试模式

* 单张图片
* 一段视频
* 文件夹（包含多张图片）

#### 参数解析

* `./tools/detect_demo.sh`文件进行分析，可知目标检测测试模式，由命令行参数`pic_mode`决定。`--pic_mode image`表示测试单张图片；`--pic_mode video`表示测试一段视频；`--pic_mode dir`表示某个文件夹下的所有图片进行目标检测。`--test_path`表示图片、视频或者文件夹路径。
* `--gpu 0`表示采用编号为0的GPU进行目标检测；`--cpu`表示屏蔽GPU，采用CPU进行目标检测；
* `--net (1) -- model (2) --cfg (3) --classes (4)` 表示对目标检测网络结构、模型权值、配置参数以及目标类别（不包含背景）的设置。

#### 目标检测（VOC2007压缩）模型下载地址

* 下载地址：萃舟服务器中`model-zoo/easy-pvanet/model/voc2007_comp/test.model`

### 3. 模型压缩

#### 命令

```shell
cd easy-pvanet
./tools/compress_model.sh
```

####  命令解析

对`./tools/compress_model.sh`文件进行分析，其命令为

```shell
python ./tools/compress_model.py --net model/voc2007/test.pt --weights model/voc2007/test.model --cfg model/voc2007/test.yml --classes model/voc2007/classes_name.txt 
```

* `--net`设置原始网络的网络结构；
* `--weights`设置原始网络的模型权值；
* `--cfg`设置原始网络的配置参数；
* `-classes`设置原始网络的目标类别（不包含背景）；
* 压缩后的模型输出在文件夹`easy-pvanet/model/comp_model`下。

### 4. 目标检测（客户端-服务器）测试

#### 命令

```shell
cd easy-pvanet
./tools/demo_server_client.sh
```

#### 命令解析

```shell
# 启动demo的服务器端
gnome-terminal -t "demo_server" -x bash -c "python ./tools/demo_server.py; exec bash;"
```

启动目标检测的服务器端

```shell
sleep 1 #睡眠1s，以便客户端能够反应过来
```

睡眠1s

```shell
# 启动demo的客户端
gnome-terminal -t "demo_client" -x bash -c "python ./tools/demo_client.py; exec bash;"
```

启动目标检测的客户端

#### 目标检测模型

`./tools/demo_server.py`中使用的目标检测模型为VOC2007压缩模型，模型路径直接在`./tools/demo_server.py`设置。

### 5. 目标精度计算

#### 命令

```shell
cd easy-pvanet
./tools/test_demo.sh
```

#### 命令解析

```shell
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
python makeMain.py --trainval 0.0 #--train 0.9
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
```

1. 如果`VOCdevkit2007`文件夹不存在，建立软链接

2. 如果未对数据集进行`trainval/test`的分配，重新分配数据集训练集和测试集的占比；如果是验证模型在新数据集上的精度，通常将`--trainval`设置为`0`即可。

3. 如果不存在文件夹`VOCdevkit2007/results/VOC2007/Main`，则重新创建以存储精度结果（目前脚本仍存在问题，并没有将结果保存至文件夹下）

4. ```shell
   # test
   python ./tools/test_demo.py --cpu --def model/voc2007_comp/test.pt --weights model/voc2007_comp/test.model --cfg model/voc2007_comp/test.yml --imdb voc_2007_test --classes data/VOCdevkit2007/classes_name.txt 2>&1 | tee log/log.txt
   ```

   * 测试脚本，可设置`cpu`模式`--cpu`，也可设置`gpu`模式和启用显卡编号`--gpu 0`
   * `--def --weights -- cfg --classes`分别设置测试模型的网络结构文件，网络权重，配置文件，检测目标类型
   * `--imdb`设置为`voc_2007_test`表示为在`VOC2007`格式数据集上的测试
   * `2>&1 | tee log/log.txt` 保存日志文件

## 中间修改

1. easy-pvanet中将ubuntu_bread项目中的lib/fast_rcnn/config.py中的(2017/01/27)

   ```
   __C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..','..', '..'))
   ```

   重新更改为

   ```
   __C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
   ```

   和原始pvanet相同



## 功能增加
### 2018/04/03

1. ~~添加模型测试`model_test`选项~~(2018/05/02已删除，测试集上的检测，可采用`test_demo.sh`完成)

   ```shell
   python ./tools/detect_demo.py --gpu 0 --net model/Always/test.prototxt --weights model/Always/Always_iter_100000.caffemodel --cfg model/Always/test.yml --classes model/Always/classes_name.txt --pic_mode model_test --test_path data/VOCdevkit2007/VOC2007 --result_dir data/result 2>&1 | tee log/log.txt
   ```

   `--pic_mode`选项在原有三个选项`image`，`video`，`dir`的基础上，添加`model_test`，并将`--test_path`指定为`data/VOCdevkit2007/VOC2007`实现模型在测试集的检测

### 2018-04-27

1. 修改`makeMain.py`文件，将`trainval,train`的比例设置提至脚本`train_demo.sh`中
2. 增加精度测试功能（需在`VOCdevkit2007/results/VOC2007/Main`文件夹下建立`results`文件夹）

### 2018-05-02

1.  完成精度计算脚本的测试`tools/test_dem0.py`和`tools/test_demo.sh`
2.  删除2018/04/03增加的测试选项`model_test`

## Bug修复（2018/04/13）

1. **修复目标框左边角点处于边界时的越界问题**

   将`easy-pvanet/lib/datasets/pascal_voc.py`215-218行

   ```python
   # Make pixel indexes 0-based
   x1 = float(bbox.find('xmin').text)-1
   y1 = float(bbox.find('ymin').text)-1
   x2 = float(bbox.find('xmax').text)-1
   y2 = float(bbox.find('ymax').text)-1
   ```

   改为

   ```python
   # Make pixel indexes 0-based
   x1 = float(bbox.find('xmin').text)
   y1 = float(bbox.find('ymin').text)
   x2 = float(bbox.find('xmax').text)
   y2 = float(bbox.find('ymax').text)
   ```

   因为一开始的`pascal_voc`数据标注起始坐标从`1`开始；而我们样本标注时，本身是从`0`开始，没必要再`-1`转换标注方式。【 `indexes 0-based`（即从0开始的标注方式）； `indexes 1-based`（即从1开始的标注方式）】

2. **修复目标检测调用`easy_detect.py/detect`函数时，阈值`CONF_THRESH`和` NMS_THRESH`设置不生效的问题**

   将`easy_detect.py`87行

   ```python
   detections = demo(net, img, _t, CLASSES, CONF_THRESH=0.7, NMS_THRESH=0.3)
   ```

   改为

   ```python
   detections = demo(net, img, _t, CLASSES, CONF_THRESH, NMS_THRESH)
   ```

   以避免阈值形参不固定死。

3. **修复`compress_model.py`运行不生效的问题**

   * 在`compress_model.py`主函数第一行增加命令

     ```python
     args = parse_args()
     ```


   * `net_weights`前后命名不一致，将`net_weight`统一修改为`net_weights`

     ```python
     net_weights = '' #line 42
     output_pt,output_weight = aux_tools.gen_merged_model(net_pt, net_weights) #line 51
     ```

4. **修复`easy_module`模块找不到的问题**

   修改`tools`文件夹下的四个`shell`执行文件，在文件的开头添加如下命令

   ```shell
   # 返回脚本文件放置的目录
   SHELL_FOLDER=$(dirname $(readlink -f "$0"))
   # 将根目录添加之PYTHONPATH中
   ROOT_PTAH=$SHELL_FOLDER/../
   export PYTHONPATH=$PYTHONPATH:$ROOT_PATH
   ```

   将`easy-pvanet`目录添加进`PYTHONPATH`中



## 感谢

* 感谢`Bug`的指出者王忠wangz@cuizhouai.com
