#!/usr/bin/env python
#-*-coding:utf-8-*-
import os.path as osp
from easy_module import aux_tools, easy_detect   
import cv2

this_dir = osp.dirname(__file__)

cfg_file = osp.join(this_dir, '..', 'model','voc2007_comp', 'test.yml')
net_pt = osp.join(this_dir, '..','model', 'voc2007_comp','test.pt')
net_weight = osp.join(this_dir, '..','model','voc2007_comp', 'test.model')
classes_path = osp.join(this_dir,'..', 'model','voc2007_comp', 'classes_name.txt')

import socket

HOST = 'localhost'
#HOST = '192.168.3.148'
PORT = 8001

if "__main__" == __name__:
  
    CLASSES = ['__background__']
    CLASSES.extend(aux_tools.get_classes(classes_path))
    sku_id = (str(ind).zfill(7) for ind in range(len(CLASSES)))
    sku_code_dist = dict(zip(CLASSES, sku_id))
    try:  
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM);  
        print("create socket succ!");  
        #setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  
        sock.bind((HOST, PORT));  
        print("bind socket succ!");  
          
        sock.listen(5);  
        print("listen succ!"); 
        # set mode
        easy_detect.set_mode('gpu',0)
        easy_detect.change_test_prototxt(net_pt, len(CLASSES))
        #load model, just load once, or else it will be very slow
        net = easy_detect.load_net(cfg_file, net_pt, net_weight) 
  
    except:  
        print("init socket err!"); 
    while True:  
        print("listen for client...");  
        conn, addr = sock.accept();  
        print("get client");  
        print(addr);  
              
        conn.settimeout(5);  
        szBuf = conn.recv(1024);    
        
        if szBuf == "1":
            # just like the predefined data structure
            results = {}
            for cls_ind, cls in enumerate(CLASSES[1:]):
                results[sku_code_dist[cls]] = 0
            #trigger the camera
            #image = recognition.take_picture(0)
            test_image_path = osp.join(this_dir, '..', 'data','demo', '001763.jpg');
            image = cv2.imread(test_image_path);
    
            #recognition : can loop
            detections = easy_detect.detect(net, image, CLASSES);
    
            #output by the predefined data structure 
            for detection in detections:
                results[sku_code_dist[detection[0]]] = results[sku_code_dist[detection[0]]] + 1;
            recognition_flag = False;
            send_content = "";
            data_content = "";
            for cls in results:
                if results[cls] != 0:
                    recognition_flag = True;
                    data_content = data_content + '{"sku_code":"%s","count":%d},' %(cls,results[cls]);

            if recognition_flag:
                data_content = data_content[0:-1]; #直接去除最后‘,’号
                send_content = '{"status":200,"msg":"Identify success!","id":1,"data":[' + data_content + ']}';
            else:
                send_content = '{"status":404,"msg":"会员未找到"}';


            #conn.send(repr(results));
            conn.send(send_content);
        elif szBuf == "0":
            conn.send("server exits!");
            break;
        else:
            #接受反馈信息，留作之后样本选取，训练
            print("recv:" + szBuf);
            send_content = '{"status":200,"msg":"Feedback success!"}';
            conn.send(send_content);
    conn.close();  
    print("end of sevice"); 
