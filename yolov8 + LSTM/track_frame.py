import cv2
from ultralytics import YOLO
import numpy as np
import math

model = YOLO("yolov8n-pose")
classNames = ['person']

cap = cv2.VideoCapture("My Video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.track(frame, persist=True)

    for r in results:
        boxes = r.boxes
        keypoints = r.keypoints
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf=math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            track_id = int(box.id[0])
            
            if cls == 0:
                class_name = classNames[cls]
                label = f'{track_id}:{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                color = (255, 0, 0)
                
                if conf > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(frame, (x1, y1), c2, color, -1)
                    cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                    
                    for keypoint in keypoints:
                        for k1, k2 in keypoint.xy[0]:
                            cv2.circle(frame, (int(k1), int(k2)), radius=2, color=(0, 255, 0), thickness=-1)

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
