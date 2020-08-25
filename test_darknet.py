from darknet import darknet
import cv2, numpy as np, sys
from ctypes import *

net = darknet.load_net(b"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/cfg/myyolov3.cfg", b"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/weight/myyolov3_final.weights", 0) 
meta = darknet.load_meta(b"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/weight/my.data") 
cap = cv2.VideoCapture(0)#"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/26-4_cam01_assault01_place01_night_spring.mp4") 
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
i = 0
while(cap.isOpened()):
    i += 1
    ret, image = cap.read()
    image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    print(i)
    if not ret: 
        break 
    frame = darknet.nparray_to_image(image)
    r = darknet.detect_image(net, meta, frame, thresh=.5, hier_thresh=.5, nms=.45, debug= False)
    print(r)
    boxes = [] 

    for k in range(len(r)): 
        width = r[k][2][2] 
        height = r[k][2][3] 
        center_x = r[k][2][0] 
        center_y = r[k][2][1] 
        bottomLeft_x = center_x - (width / 2) 
        bottomLeft_y = center_y - (height / 2) 
        x, y, w, h = bottomLeft_x, bottomLeft_y, width, height 
        boxes.append((x, y, w, h))

    for k in range(len(boxes)): 
        x, y, w, h = boxes[k] 
        top = max(0, np.floor(x + 0.5).astype(int)) 
        left = max(0, np.floor(y + 0.5).astype(int)) 
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int)) 
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int)) 
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2) 
        cv2.line(image, (top + int(w / 2), left), (top + int(w / 2), left + int(h)), (0,255,0), 3) 
        cv2.line(image, (top, left + int(h / 2)), (top + int(w), left + int(h / 2)), (0,255,0), 3) 
        cv2.circle(image, (top + int(w / 2), left + int(h / 2)), 2, tuple((0,0,255)), 5)

    cv2.imshow('frame', image)

    darknet.free_image(frame) ## darknet에서 쓰는 c언어 동적할당 해제해주는 함수(써주어야 메모리가 버틴다.)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
