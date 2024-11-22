자동차 번호인식
import cv2
#opencv 라이브러리
import numpy as np
#numpy 라이브러리
import matplotlib.pyplot as plt
#결과물 출력을 위한 matplot라이브러리
import pytesseract
#글씨를 읽어내기 위한 pytesseract 라이브러리
#plt.style.use('dark_background')
cap = cv2.VideoCapture(0)
#웹캠 호출cap.set(3, 320)
cap.set(4, 240)
#320*240의 사이즈 지정
while True:
  #무한반복    
  ret, frame = cap.read()    
  #영상 사진을 불러온다    
  cv2.imshow('webcam', frame)    
  #화면 출력    
  if cv2.waitKey(1) == ord('q'):    
    #q를 누르면        
    image = frame        
    #영상 사진 저장        
    height, width, channel = image.shape        
    #높이, 너비, 채널 정하기        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        
    #그레이 스케일 변환        
    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)        
    #가우시안블러로 노이즈 없애기        
    img_thresh = cv2.adaptiveThreshold(        
        #이미지 threshold 처리            
        img_blurred,            
        #블러            
        maxValue=255.0,            
        #threshold 최대값            
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,            
        #검은색과 흰색으로 이미지 나누기            
        thresholdType=cv2.THRESH_BINARY_INV,            
        #threshold타입            
        blockSize=19,            
        #threshold값을 계산하기 위해 사용되는 블록 크기            
        C=9            
        #계산된 평균으로부터 뺄 상수값        
        )        
        _, contours, _ = cv2.findContours(        
            #윤곽선 찾기            
            img_thresh,            
            #threshold한 이미지            
            mode = cv2.RETR_LIST,            
            #윤곽 검색 모드 설정            
            method=cv2.CHAIN_APPROX_SIMPLE            
            #윤곽 근사방법        
        )        
        temp_result = np.zeros((height, width, channel), dtype=np.uint8)        
        #크기가 height, width, channel인 배열 생성        
        contours_dict = []        
        #배열선언        
        for contour in contours:        
          #for 문            
          x, y, w, h = cv2.boundingRect(contour)            
          #윤곽선을 감싸는 사각형 구하기            
          #x값, y값, 너비, 높이            
          cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255,255,255), thickness=2)            
          #사각형을 그린다            
          contours_dict.append({            
              #배열에 값들을 각각 저장                
              'contour': contour,                
              #contour값                
              'x': x,                
              #x값                
              'y': y,                
              #y값                
              'w': w,                
              #너비값                
              'h': h,                
              #높이값                
              'cx' : x + (w/2),                
              #중심좌표                
              'cy' : y + (h/2)                
              #중심좌표            
              })        
          MIN_AREA = 60        
          #bounding rect의 최소 넓이        
          MIN_WIDTH, MIN_HEIGHT = 2, 8        
          #최소 높이와 너비        
          MIN_RATIO, MAX_RATIO = 0.25, 1.0        
          #가로 대비 세로 비율의 최소 최대 값        
          possible_contours = []        
          #조건 만족한 번호판을 저장하는 배열        
          cnt = 0        
          #변수        
          for d in contours_dict:        
            #for 문           
            area = d['w'] * d['h']            
            #넓이            
            ratio = d['w'] / d['h']            
            #가로 대비 세로 비율            
            if area > MIN_AREA \                
              and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \                
              and MIN_RATIO < ratio < MAX_RATIO:                
              #조건과 비교                
              d['idx'] = cnt                
              #인덱스 저장                
              cnt += 1                
              #변수 + 1                
              possible_contours.append(d)                
              #배열에 저장        
              temp_result = np.zeros((height, width, channel), dtype=np.uint8)        
              #크기가 height, width, channel인 배열 생성        
              for d in possible_contours:            
                #cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))            
                cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),                          
                              thickness=2)            
                #배열에 들어간 번호판 후보들 그리기        
                MAX_DIAG_MULTIPLYER = 5        
                #사각형과 사각형 사이의 거리 제한        
                #각 사각형들의 중심점 거리        
                MAX_ANGLE_DIFF = 90.0        
                #중심점들이 이루는 각의 최대크기        
                MAX_AREA_DIFF = 0.5        
                #면적차이        
                MAX_WIDTH_DIFF = 0.8        
                #너비차이        
                MAX_HEIGHT_DIFF = 0.2        
                #높이차이        
                MIN_N_MATCHED = 3        
                #위 조건을 만족하는 사각형들이 3개 이하일때 제외        
                def find_chars(contour_list):        
                  #함수 선언            
                  matched_result_idx = []            
                  #최종 결과물의 인덱스 저장            
                  for d1 in contour_list:            
                    #for문 1                
                    matched_contours_idx = []                
                    #조건에 맞는 결과 저장                
                    for d2 in contour_list:               
                    #for문 2                    
                    if d1['idx'] == d2['idx']:                    
                      #contour가 같으면 똑같은 사각형이므로 비교x                        
                      continue                        
                      #넘기기                    
                      dx = abs(d1['cx'] - d2['cx'])                    
                      #중심점들의 가로 길이                    
                      dy = abs(d1['cy'] - d2['cy'])                    
                      #중심점들의 세로 길이                    
                      diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)                   
                      #d1사각형의 대각길이                    
                      distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))                    
                      #벡터a와 벡터b의 거리 구하기                    
                      if dx == 0:                    
                        #x값이 같을때                        
                        angle_diff = 90                        
                        #90으로 준다                        
                        #예외처리                    
                        else:                        
                          angle_diff = np.degrees(np.arctan(dy / dx))                        
                          #아크 탄젠트 값을 구한다                        
                          #'라디안'을 '도'로 변경                    
                          area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])                    
                          #면적의 비율                    
                          width_diff = abs(d1['w'] - d2['w']) / d1['w']                    
                          #너비의 비율                    
                          height_diff = abs(d1['h'] - d2['h']) / d1['h']                    
                          #높이의 비율                    
                          if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \                    
                            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \                    
                            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:                    
                            #기준에 맞다면                        
                            matched_contours_idx.append(d2['idx'])                        
                            #결과물의 d2 인덱스만 저장                
                            matched_contours_idx.append(d1['idx'])                
                            #결과물의 d1 인덱스 저장                
                            if len(matched_contours_idx) < MIN_N_MATCHED:                
                              #번호판 후보의 윤곽선 개수가 3보다 작으면                
                              #번호판은 7글자이므로                    
                              continue                    
                              #제외                
                              matched_result_idx.append(matched_contours_idx)                
                              #최종 후보에 넣어준다                
                              unmatched_contour_idx = []                
                              #후보에서 제외된 사각형들을 넣어주는 공간                
                              for d4 in contour_list:                    
                                if d4['idx'] not in matched_contours_idx:                    
                                  #후보가 아니라면                        
                                  unmatched_contour_idx.append(d4['idx'])                        
                                  #리스트에 넣어준다                
                                  unmatched_contour = np.take(possible_contours, unmatched_contour_idx)                
                                  #unmatched_contour_idx에서 possible_contours와 같은 인덱스만 추출                
                                  recursive_contour_list = find_chars(unmatched_contour)                
                                  #함수에 넣어서 재귀함수로 돌린다                
                                  for idx in recursive_contour_list:                
                                    #재귀함수로 걸러진 후보들로 for문 돌리기                    
                                    matched_result_idx.append(idx)                    
                                    #살아남은 후보들을 추가해준다                
                                    break                
                                    #정지            
                                    return matched_result_idx            
                                    #값 반환        
                                    result_idx = find_chars(possible_contours)        
                                    #결과물들을 함수에 넣어서 돌린다        
                                    matched_result = []        
                                    #결과물 저장        
                                    for idx_list in result_idx:            
                                      matched_result.append(np.take(possible_contours, idx_list))            
                                      #결과물들을 주어진 index로 정렬        
                                      temp_result = np.zeros((height, width, channel), dtype=np.uint8)        
                                      #크기가 height, width, channel인 배열 생성        
                                      for r in matched_result:            
                                        for d in r:                
                                          #cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))                
                                          cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),                              
                                                        thickness=2)                
                                          #후보들을 다시 그려본다        
                                          PLATE_WIDTH_PADDING = 1.3        
                                          #너비 패딩값        
                                          PLATE_HEIGHT_PADDING = 1.5        
                                          #높이패딩값        
                                          MIN_PLATE_RATIO = 3        
                                          #최소값        
                                          MAX_PLATE_RATIO = 10        
                                          #최대값        
                                          plate_imgs = []        
                                          #배열선언        
                                          plate_infos = []        
                                          #배열선언        
                                          for i, matched_chars in enumerate(matched_result):        
                                            #최종 후보들에 대해 루프 돌림            
                                            sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])           
                                            #x방향에 순차적으로 정렬            
                                            plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2            
                                            #센터x좌표            
                                            plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2            
                                            #센터y좌표            
                                            plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING            
                                            #너비 구하기            
                                            sum_height = 0            
                                            #높이의 합            
                                            for d in sorted_chars:                
                                              sum_height += d['h']   
                                              #정렬한 후보들의 높이의 합을 구한다            
                                              plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)            
                                              #높이 구하기            
                                              triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']            
                                              #삼각형의 높이 구하기            
                                              triangle_hypotenus = np.linalg.norm(            
                                                  #삼각형의 빗변 구하기                
                                                  np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -                
                                                  #첫번째 사각형의 중심 좌표                
                                                  np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])                
                                                  #마지막 사각형의 중심 좌표            )            
                                                  angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))            
                                                  #아크사인으로 빗변분의 높이의 라디안 값을 구하고 '도'로 변환한다            
                                                  rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)            
                                                  #로테이션 매트릭스를 구한다            
                                                  img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))            
                                                  #이미지 변형            
                                                  img_cropped = cv2.getRectSubPix(            
                                                      #회전된 이미지에서 원하는 부분만 잘라낸다                
                                                      img_rotated,                
                                                      #이미지 변형               
                                                      patchSize=(int(plate_width), int(plate_height)),                
                                                      center=(int(plate_cx), int(plate_cy))                
                                                      #번호판 부분만 잘라낸다            
                                                      )            
                                                  if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[                
                                                      0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:                
                                                  continue   
                                                  #조건과 비교하여 통과            
                                                  plate_imgs.append(img_cropped)            
                                                  #img_cropped 추가            
                                                  plate_infos.append({            
                                                      #plate_infos에 각각의 정보 추가               
                                                      'x': int(plate_cx - plate_width / 2),                
                                                      #x값           
                                                      'y': int(plate_cy - plate_height / 2),          
                                                      #y값           
                                                      'w': int(plate_width),        
                                                      #너비값             
                                                      'h': int(plate_height)     
                                                                 #높이값      
                                                            })   
                                                       longest_idx, longest_text = -1, 0   
                                                       plate_chars = []     
                                                     for i, plate_img in enumerate(plate_imgs):    
                                                        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)      
                                                             _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)     
                                                                    #threshold       
                                                       _, contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)    
                                                               #contours 찾기        
                                                      plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]      
                                                        plate_max_x, plate_max_y = 0, 0     
                                                         for contour in contours:           
                                                       x, y, w, h = cv2.boundingRect(contour)           
                                                            #boundingRect 구하기          
                                                        area = w * h       
                                                           #면적 구하기        
                                                          ratio = w / h         
                                                         #비율 구하기         
                                                         if area > MIN_AREA \  
                                                                and w > MIN_WIDTH and h > MIN_HEIGHT \        
                                                          and MIN_RATIO < ratio < MAX_RATIO:       
                                                           #기준에 맞는지 체크            
                                                          if x < plate_min_x:              
                                                            plate_min_x = x              
                                                        if y < plate_min_y:                  
                                                        plate_min_y = y              
                                                        if x + w > plate_max_x:        
                                                                  plate_max_x = x + w              
                                                        if y + h > plate_max_y:                
                                                          plate_max_y = y + h           
                                                           #번호판의 x, y의 최대 최소값 구하기       
                                                       img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]     
                                                         #번호판 부분만 잘라내기        
                                                      img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)   
                                                               #노이즈 없애기       
                                                       _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)     
                                                              #threshold         
                                                     img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, 
                                                                                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))      
                                                           #이미지에 패딩을 준다        
                                                      chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 0')       
                                                           #이미지에서 글자 읽기        
                                                      result_chars = ''           
                                                   #정답글자          
                                                    has_digit = False        
                                                      for c in chars:         
                                                         if ord('가') <= ord(c) <= ord('힣') or c.isdigit():         
                                                       #숫자나 한글이 포함되어 있는지 체크                 
                                                     if c.isdigit():                   
                                                 #숫자가 한개라도 있는지 체크                 
                                                  has_digit = True                    
                                                  result_chars += c               
                                                  #최종 결과물에 추가           
                                                  print(result_chars)         
                                                     #최종 결과물 출력         
                                                     plate_chars.append(result_chars)         
                                                        if has_digit and len(result_chars) > longest_text:      
                                                        #가장 긴 문자열 구하기             
                                                     longest_idx = i            
                                                      #번호판으로 저장
