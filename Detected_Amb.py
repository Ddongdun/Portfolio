import numpy as np 
#라이브러리 임포트
import os
import threading
import tensorflow as tffrom matplotlib
import pyplot as pltfrom PIL 
import Imagefrom PIL 
import ImageFileImageFile.LOAD_TRUNCATED_IMAGES = Truefrom object_detection.utils 
import ops as utils_opsfrom utils 
import label_map_utilfrom utils 
import visualization_utils as vis_utilfrom socket 
import *from sys 
import exitimport 
socketHOST = '192.168.0.9' 
#서버와 연결할 소켓
PORT = 8602
so = socket.socket(AF_INET, SOCK_STREAM)
def amb(): 
  #소스 전체를 임포트 하기 위해 함수로 묶어줌    
  PATH_TO_CKPT = 'D:/object/export_dir/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb' 
  #내가 교육한 모델    
  PATH_TO_LABELS = os.path.join('D:/object/label_map.pbtxt')    
  NUM_CLASSES = 3 
  #SUV, 세단, 구급차    
  detection_graph = tf.Graph() with detection_graph.as_default(): 
  #그래프에 노드 추가      
  od_graph_def = tf.GraphDef() with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid: 
  #모델을 불러옴        
  serialized_graph = fid.read()        
  od_graph_def.ParseFromString(serialized_graph)        
  tf.import_graph_def(od_graph_def, name='')    
  label_map = label_map_util.load_labelmap(PATH_TO_LABELS) 
  #라벨을 불러옴    
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)    
  category_index = label_map_util.create_category_index(categories)    
def load_image_into_numpy_array(image): 
  #사진을 넘파이 배열로 불러옴      
  (im_width, im_height) = image.size      
  return np.array(image.getdata()).reshape(          (im_height, im_width, 3)).astype(np.uint8)        
  try: 
  #사진이 들어오면        
    so.connect((HOST, PORT)) 
    #소켓 연결       
     PATH_TO_TEST_IMAGES_DIR = 'D:/deeplearning/models-master/research/object_detection/test_images'        
     #TEST_IMAGE_PATHS = os.listdir(PATH_TO_TEST_IMAGES_DIR)        
     TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image.jpg')]
      #image.jpg를 불러온다        IMAGE_SIZE = (12, 8)        
      def run_inference_for_single_image(image, graph): 
        #불러온 이미지에 대해 실행          
        with graph.as_default():            
          with tf.Session() as sess:              
            # Get handles to input and output tensors              
            ops = tf.get_default_graph().get_operations()              
            all_tensor_names = {output.name for op in ops for output in op.outputs}              
            tensor_dict = {}              
            for key in [ 
                #각 키 값에 대해서                  
                'num_detections', 'detection_boxes', 'detection_scores',                  
                'detection_classes', 'detection_masks' 
                #이름을 지정              
                ]:                
                tensor_name = key + ':0'                
                if tensor_name in all_tensor_names:                  
                  tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)              
                  if 'detection_masks' in tensor_dict:                
                    # The following processing is only for single image                
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0]) 
                    #사이즈가 1인 차원을 제거                
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])                
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.                
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32) 
                    #새로운 형태로 바꾸어줌                
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    #특정 부분 추출                
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])                
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(                    
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])               
                     detection_masks_reframed = tf.cast(                    
                         tf.greater(detection_masks_reframed, 0.5), tf.uint8)                
                     # Follow the convention by adding back the batch dimension                
                     tensor_dict['detection_masks'] = tf.expand_dims(                    
                         detection_masks_reframed, 0)              
                     image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')              
                     # Run inference              
                     output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})              
                     # all outputs are float32 numpy arrays, so convert types as appropriate              
                     output_dict['num_detections'] = int(output_dict['num_detections'][0])              
                     output_dict['detection_classes'] = output_dict[                  
                         'detection_classes'][0].astype(np.uint8)              
                         output_dict['detection_boxes'] = output_dict['detection_boxes'][0]              
                         output_dict['detection_scores'] = output_dict['detection_scores'][0]              
                         if 'detection_masks' in output_dict:                
                          output_dict['detection_masks'] = output_dict['detection_masks'][0]          
                          return output_dict        for image_path in TEST_IMAGE_PATHS: 
                          #이미지에 대해 실행          
                          image = Image.open(image_path) 
                          #이미지를 불러옴          
                          image_np = load_image_into_numpy_array(image) 
                          #이미지를 넘파이 배열로 바꿈          
                          image_np_expanded = np.expand_dims(image_np, axis=0)          
                          output_dict = run_inference_for_single_image(image_np, detection_graph)          
                          vis_util.visualize_boxes_and_labels_on_image_array(              
                              image_np, 
                              #이미지에 대해서              
                              output_dict['detection_boxes'], 
                              #테두리 상자              
                              output_dict['detection_classes'], 
                              #추측한 종류              
                              output_dict['detection_scores'], 
                              #정확도              
                              category_index,              
                              instance_masks=output_dict.get('detection_masks'),              
                              use_normalized_coordinates=True,              
                              line_thickness=8)          
                          plt.figure(figsize=IMAGE_SIZE)          
                          plt.imshow(image_np)          
                          print ('img',image_path)          
                          plt.savefig(image_path[:-3]+'png')        
                          add = output_dict['detection_classes']        
                          print(add[0])        
                          if(add[0] == 1): 
                            #만약 구급차로 추측되었다면            
                            print("구급차 입니다")            
                            so.send('1*1*E'.encode()) 
                            #서버로 값 전송        
                            else:            
                              print("구급차가 아닙니다") 
                              #구급차가 아니라면            
                              so.send('1*1*N'.encode()) 
                              #서버로 값 전송    
                              except: 
                                #사진을 불러오지 못했을 때        
                                print("예외 처리")
