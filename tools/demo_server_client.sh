#!/bin/bash
# add root_dir into PYTHONPATH
# 返回脚本文件放置的目录
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
# 将根目录添加之PYTHONPATH中
ROOT_PTAH=$SHELL_FOLDER/../
export PYTHONPATH=$PYTHONPATH:$ROOT_PATH

# 启动demo的服务器端
gnome-terminal -t "demo_server" -x bash -c "python ./tools/demo_server.py; exec bash;"

sleep 1 #睡眠1s，以便客户端能够反应过来

# 启动demo的客户端
gnome-terminal -t "demo_client" -x bash -c "python ./tools/demo_client.py; exec bash;"


# gnome-terminal -t "title-name" -x bash -c "sh ./run.sh;exec bash;"
# -t 为打开终端的标题，便于区分。
# -x 后面的为要在打开的终端中执行的脚本，根据需要自己修改就行了。
# 最后的exec bash;是让打开的终端在执行完脚本后不关闭。
