클라이언트
# -*- coding: utf-8 -*-
import socket
import json
import cv2
import sys
import time
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('192.168.0.30', 8801))
# img_path = "./evalimg/img_" + time + ".jpg"
img_path = "/home/pi/image.jpg"
cam = cv2.VideoCapture(0)
ret, frame = cam.read()
cv2.imwrite(img_path, frame)
cam.release()
#cv2.destroyAllWindows()
sock.send(img_path.encode())
image = open(img_path, 'rb')
image_send = image.read(1048576)
print(image_send)
print(type(image_send))
print(sys.getsizeof(image_send))
sock.send(str(sys.getsizeof(image_send)).encode())
time.sleep(10)
# sock.send(input_data.encode())
sock.send((image_send.decode() + "\n").encode())
data = sock.recv(65536).decode('utf-8')
data = json.loads(data)
print("데이터를 돌려받았다 : ", data)
