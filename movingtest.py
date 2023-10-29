from ultralytics import YOLO
import cv2 
import math 
import numpy as np
import glob
import os
import shutil
from matplotlib import pyplot as plt
import torch
from collections import deque


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# models
model = YOLO("yolov8m.pt")
model_cw = YOLO("crosswalk_model/best.pt")
# Set the device for inference (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model_cw = model_cw.to(device)

# # object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]




#Bounding Box Merge Algorithm
def merge_bounding_boxes(bounding_boxes,image_width,image_height):
  [x,y] = [image_height,image_width]
  [w,h] = [0,0]
  for i in range(0,bounding_boxes.shape[0]):
    if(bounding_boxes.ndim == 1):
      center_x_1 = int(bounding_boxes[0]*image_width)
      center_y_1 = int(bounding_boxes[1]*image_height) 
      width_1 = int(bounding_boxes[2]*image_width) 
      height_1 = int(bounding_boxes[3]*image_height)
      x = int(center_x_1 - (width_1/2))
      y = int(center_y_1 - (height_1/2))
      w = int(center_x_1 + (width_1/2))
      h = int(center_y_1 + (height_1/2))
    else:
      center_x_1 = int(bounding_boxes[i,0]*image_width)
      center_y_1 = int(bounding_boxes[i,1]*image_height) 
      width_1 = int(bounding_boxes[i,2]*image_width) 
      height_1 = int(bounding_boxes[i,3]*image_height)
      vert_x = int(center_x_1 - (width_1/2))
      vert_y = int(center_y_1 - (height_1/2))
      vert_w = int(center_x_1 + (width_1/2))
      vert_h = int(center_y_1 + (height_1/2))
      if(vert_x < x):
        x = vert_x
      if(vert_y < y):
        y = vert_y
      if(vert_w > w):
        w = vert_w
      if(vert_h > h):
        h = vert_h
    return np.array([x,y,w,h])

video = 'video.mp4' #@param {type:"string"}

#car detection
result = model.predict('detection/video', imgsz=512, save=False, save_txt=True, save_conf=True, project="video_results/")
path = os.getcwd()+'/video_results/predict/labels/'
video_path = os.getcwd()+'/detection/video/'
save_path = os.getcwd()+'/video_results/1/'
frames_path = os.getcwd()+'/video_results/predict/frames/'
video_name = video.replace('.mp4', '')


# Create the frames directory if it doesn't exist
os.makedirs(frames_path, exist_ok=True)

print('Reading Video: '+video_path+video)
v = cv2.VideoCapture(video_path+video)
frame_count = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
fps = v.get(cv2.CAP_PROP_FPS)
video_w = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('Video Info' + '  FPS: '+str(int(fps))+' Width: '+str(video_w)+' Height: '+str(video_h))

frames = None
frame_prev = None
threshold = 4000  # Adjust this threshold as needed


motion_state_queue = deque(maxlen=5)
movement = True

for filename in glob.glob(os.path.join(path, '*.txt')):
    with open(os.path.join(os.getcwd(), filename), 'r') as f:
        bounding_boxes = None
        confidences = None
        # Saves the Frame number in an array
        frame_num = int(os.path.basename(filename).replace('.txt', '').replace(video_name + '_', ''))
        frames = (np.vstack((frames, frame_num)) if (frames is not None) else frame_num)
        class_labels = []
        for line in f:
            cl, label_x, label_y, label_w, label_h, conf = line.split(' ')
            class_labels.append(int(cl))  # Store class labels
            b = float(conf)
            a = np.array([float(label_x), float(label_y), float(label_w), float(label_h)])
            bounding_boxes = (np.vstack((bounding_boxes, a)) if (bounding_boxes is not None) else a)
            confidences = (np.vstack((confidences, b)) if (confidences is not None) else b)
        conf_max = np.amax(confidences)
        
        # Get Frames from Video
        v.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        res, image = v.read()
        # cv2_imshow(image)
        image_height = video_h
        image_width = video_w

        for i in range(len(class_labels)):
            class_id = class_labels[i]
            class_name = classNames[class_id] if class_id < len(classNames) else f'Class {class_id}'

            if confidences[i] > 0.5:  # Customize confidence threshold as needed
                [x, y, w, h] = merge_bounding_boxes(bounding_boxes[i:i+1], image_width, image_height)
                roi = image#[y:h, x:w]
            if roi is not None: 
              # print(roi.shape)

              gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

              if frame_prev is None:
                  frame_prev = gray#[y:h, x:w]
                  # print(frame_prev.shape)
                  is_moving = 0  # Assume stationary for the first frame
              else:
                  # print(frame_prev.shape)
                  # frame_prev = frame_prev[y:h, x:w]
                  frame_diff = cv2.absdiff(frame_prev, gray)
                  motion_area = cv2.countNonZero(frame_diff)

                  if motion_area > threshold:
                      is_moving = 1
                  else:
                      is_moving = 0
                  motion_state_queue.append(is_moving)
                  moving_average = sum(motion_state_queue) / len(motion_state_queue)

                  if moving_average>=0.3:
                     movement = True
                  else:
                     movement = False

                  frame_prev = gray

                  color = (0, 255, 0) if movement else (0, 0, 255)


                  # Create Rect
                  cv2.rectangle(image, (x, y), (w, h), color, 4)
                  cv2.putText(image, f'{classNames[class_id]} {conf_max:.2f}{" (Moving)" if movement else "(Stationary)"}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


          # [x, y, w, h] = merge_bounding_boxes(bounding_boxes, image_width, image_height)
          # cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 4)
          # cv2.putText(image, 'crosswalk ' + "%.2f" % conf_max, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if image is None:
            continue
        cv2.imwrite(frames_path + str(frame_num) + '.jpg', image)
v.release()
print("Generating video for car-detection")


#SAVING NEW VIDEO INFERENCES
cap = cv2.VideoCapture(video_path+video)
vid_writer = cv2.VideoWriter(save_path+video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (video_w, video_h))
frame_num = -1
while (cap.isOpened()):
  frame_num +=1
  ret, frame = cap.read()
  if ret == True:
    if(frame_num in frames):
      frame = cv2.imread(frames_path+str(frame_num)+'.jpg')
    vid_writer.write(frame)
  else:
    break
cap.release()
vid_writer.release()

shutil.rmtree(frames_path)
print('Result Saved on '+save_path+video)



###### crosswalk detection


result_cw = model_cw.predict(save_path, imgsz=512, save=False, save_txt=True, save_conf=True, project="video_results/")
path_cw = os.getcwd()+'/video_results/predict2/labels/'
video_path_cw = os.getcwd()+'/video_results/1/'
frames_path_cw = os.getcwd()+'/video_results/predict2/frames/'
video_name_cw = video.replace('.mp4', '')
save_path_cw = os.getcwd()+'/video_results/'

os.makedirs(frames_path_cw, exist_ok=True)


print('Reading video for crosswalk detection: '+video_path_cw+video)
v = cv2.VideoCapture(video_path_cw+video)
frame_count_cw = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
fps = v.get(cv2.CAP_PROP_FPS)
video_w = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('Video Info' + '  FPS: '+str(int(fps))+' Width: '+str(video_w)+' Height: '+str(video_h))

frames = None
for filename in glob.glob(os.path.join(path_cw, '*.txt')):
  with open(os.path.join(os.getcwd(),filename), 'r') as f:
    bounding_boxes = None
    confidences = None
    #Saves the Frame number in an array
    frame_num = int(os.path.basename(filename).replace('.txt', '').replace(video_name_cw +'_',''))
    frames = (np.vstack((frames, frame_num)) if (frames is not None) else frame_num)
    for line in f:
      cl, label_x, label_y, label_w, label_h, conf = line.split(' ')
      b = float(conf)
      a = np.array([float(label_x),float(label_y),float(label_w),float(label_h)])
      bounding_boxes = (np.vstack((bounding_boxes, a)) if (bounding_boxes is not None) else a)
      confidences = (np.vstack((confidences, b)) if (confidences is not None) else b)
    conf_max = np.amax(confidences)
    #Get Frames from Video
    v.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    res, image = v.read()
    #cv2_imshow(image)
    image_height = video_h
    image_width = video_w
    [x,y,w,h] = merge_bounding_boxes(bounding_boxes, image_width, image_height)
    cv2.rectangle(image, (x,y), (w,h), (255,0,0), 4)
    cv2.putText(image, 'crosswalk ' + "%.2f" % conf_max, (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    if image is None:
      continue
    cv2.imwrite(frames_path_cw+str(frame_num)+'.jpg', image)
v.release()
print("Crosswalk detection Complete. Generating Video output.")


#SAVING NEW VIDEO INFERENCES
cap = cv2.VideoCapture(video_path_cw+video)
vid_writer = cv2.VideoWriter(save_path_cw+video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (video_w, video_h))
frame_num = -1
while (cap.isOpened()):
  frame_num +=1
  ret, frame = cap.read()
  if ret == True:
    if(frame_num in frames):
      frame = cv2.imread(frames_path_cw+str(frame_num)+'.jpg')
    vid_writer.write(frame)
  else:
    break
cap.release()
vid_writer.release()

shutil.rmtree(frames_path_cw)
print('Result Saved on '+save_path_cw+video)
