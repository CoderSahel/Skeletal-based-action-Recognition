from ultralytics import YOLO
import cv2


#model_path = ''

image_path = './img.jpg'
img = cv2.imread(image_path)

model = YOLO('yolov8n-pose.pt')

results = model(image_path)

for result in results:
    #'''
    for keypoint_indx, keypoint in enumerate(result.keypoints.data):
        print("here", keypoint[0], keypoint[1])
        
        #cv2.putText(img, str(keypoint_indx), (int(keypoint[0]), int(keypoint[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #    '''
    #print(result.keypoints)
    for keypoint_indx, keypoint in enumerate(result.keypoints.data):
        print("keypoint_indx : ", keypoint_indx)
        print("keypoint : ", keypoint)

cv2.imshow('img', img)
cv2.waitKey(0)