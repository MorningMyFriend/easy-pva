#!/usr/bin/env python  
  
import socket; 

HOST = 'localhost'
#HOST = '192.168.3.148'
PORT = 8001 
  
if "__main__" == __name__:  
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM);  
    #setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.connect((HOST, PORT));  
    sock.send('1');  
  
    szBuf = sock.recv(1024);  
    print("recv " + szBuf);
    # add content
    sock.send('see you again!');

    sock.close();  
    print("end of connect"); 
