print("Import...")
import socketimport datetime as dt
import osimport threading
import json
import sysimport time as tfrom obyy 
import ambprint("Import Finish!")
def threaded(client_socket): 
  #사진을 받는 쓰레드 함수    
  while True:        
    print("대기중")       
    data = client_socket.recv(65536)        
    print(type(data))        
    print(data)        
    # image_len = int(client_socket.recv(1024).decode())        
    image_len = int(client_socket.recv(1024))        
    print(image_len)        
    total_image = b''        
    total_len = 0        
    while True:            
      print("total -> ", total_len)            
      if total_len >= image_len:                
        break            
        print("이미지 받는중...")            
        image_data = client_socket.recv(image_len)            
        print("buf -> ", sys.getsizeof(image_data))            
        if sys.getsizeof(image_data) == 33:                
          break            
          total_len += sys.getsizeof(image_data)            
          total_image += image_data        
          print("이미지 전송 끝")        
          #time = dt.datetime.now().strftime("%y%m%d%H%M%S")        
          image_path = 'D:/deeplearning/models-master/research/object_detection/test_images/image.jpg'        
          try:            
            image = open(image_path, 'wb')            
            image.write(total_image)       
            except FileNotFoundError:            
              print("파일을 찾을 수 없습니다.")        
              if os.path.isfile(image_path):            
                amb()            
                # ddddddd        
                else:            
                  client_socket.send("Fail".encode())        
                  # print("recieve Data : ", data.decode())    
                  #server_socket.close()def Main(): 
                  #서버를 바인딩하는 메인 함수    
                  server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    
                  # server_socket.bind(('127.0.0.1', 10117))    
                  server_socket.bind(('192.168.0.30', 8801))    
                  server_socket.listen(5)    
                  print("socket is listening")    
                  while True:        
                    client_socket, addr = server_socket.accept()        
                    ip, port = str(addr[0]), str(addr[1])        
                    print("Connected with " + ip + ":" + port)        
                    threading.Thread(target=threaded, args=(client_socket, )).start()
                    if __name__ == 'main':    Main()
