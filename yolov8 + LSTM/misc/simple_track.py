from ultralytics import YOLO
import cv2
import math
import numpy as np

def find_midpoint(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)
# load yolov8 model
model = YOLO('yolov8n.pt')
classNames = ['person']
# load video
#video_path = './test.mp4'
cap = cv2.VideoCapture("My Video.mp4")
#cv2.namedWindow("Display1", cv2.WINDOW_NORMAL) 

ret = True
# read frames
while ret:
    ret, frame = cap.read()
    if ret:

        # detect objects
        # track objects
        results = model.track(frame, persist=True)
        for r in results:
                boxes=r.boxes
                for box in boxes:
                    x1,y1,x2,y2=box.xyxy[0]
                    x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                    print(x1,y1,x2,y2)
                    conf=math.ceil((box.conf[0]*100))/100
                    cls=int(box.cls[0])
                    if cls==0:
                        class_name=classNames[cls]
                        label=f'{class_name}{conf}'
                        t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                        print(t_size)
                        c2 = x1 + t_size[0], y1 - t_size[1] - 3
                        color=(0, 204, 255)
                        FONTS = cv2.FONT_HERSHEY_COMPLEX
                        if conf>0.5:
                            try:
                                    frame = results[0].plot()
                                    midpoint = find_midpoint(x1, y1, x2, y2)
                                    #image = np.zeros((500, 500, 3), dtype=np.uint8)
                                    _,_,screen_width, screen_height = cv2.getWindowImageRect('frame')
                                    midpoint_screen = ((screen_width+1) // 2, screen_height)
                                    #cv2.line(frame,(((screen_width+1) // 2), midpoint[1]),midpoint, (255, 255, 255), 2)
                                    dis=abs((screen_width+1) // 2 - midpoint[0])
                                    val2=midpoint[0]
                                    print(val2)
                                    val3=midpoint[1]
                                    print(val3)
                                    hypo=math.sqrt(math.pow(val2,2)+math.pow(val3,2))
                                    print("Hypo",hypo)
                                    print("dis",dis)
                                    val=dis/hypo
                                    val1=int(math.degrees(math.asin(val)))
                                    print("value:",val1)
                                    cv2.putText(frame, str(val1)[0:5]+"  "+str(hypo*5/1000), (x1+170,y1-8),FONTS, 0.48,(0, 0, 0), 2)
                            except ValueError:
                                pass
                            # cv2.rectangle     
        # cv2.putText
    

        # visualize
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break