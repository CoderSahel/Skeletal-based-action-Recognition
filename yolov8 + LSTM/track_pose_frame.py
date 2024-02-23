import cv2
from ultralytics import YOLO
import math

def render_text(frame, label, x1, y1, c2):
    cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
    cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

def main():
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
            
            for box, keypoints in zip(boxes, keypoints):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0]*100))/100
                cls = int(box.cls[0])
                track_id = int(box.id[0])
                
                if cls == 0 and conf > 0.5:
                    label = f'{track_id}:person {conf}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    render_text(frame, label, x1, y1, c2)
                    
                    for k1, k2 in keypoints.xy[0]:
                        cv2.circle(frame, (int(k1), int(k2)), radius=2, color=(0, 255, 0), thickness=-1)
                    
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
