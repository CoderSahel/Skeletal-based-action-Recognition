import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")  # Adjust path if needed

# Initialize webcam
cap = cv2.VideoCapture("dataset\walk\w1.mp4")

while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    # Perform pose estimation using YOLOv8
    results = model(frame)

    # Draw bounding boxes and pose estimation lines
    for result in results:
        x1, y1, x2, y2 = map(int, result.boxes.xyxy[0])

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Extract keypoints
        keypoints = result.keypoints.xy[1][0][:17].tolist()  # 17 keypoints for pose

        # Draw pose estimation lines
        for i in range(0, len(keypoints), 2):
            x, y = keypoints[i], keypoints[i + 1]
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            """if i < 14:  # Connect adjacent keypoints
                cv2.line(frame, (int(x), int(y)), (int(keypoints[i + 2]), int(keypoints[i + 3])), (0, 0, 255), 2)"""

    # Display the frame with pose estimation
    cv2.imshow("Pose Estimation", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
