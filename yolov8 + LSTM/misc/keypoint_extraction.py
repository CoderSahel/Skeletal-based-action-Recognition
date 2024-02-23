from ultralytics import YOLO

# Load a pretrained YOLOv8n-pose Pose model
model = YOLO('yolov8n-pose.pt')

# Run inference on an image
results = model(source="My Video.mp4", show=True, conf=0.7, stream=True)  # results list

# View results
for r in results:
    print(r.keypoints.xyn)
    file = open('geek.txt','a')
    file.write(str(r.keypoints.xy[0][0].numpy()))
    file.write("\n\n")
    file.close()

"""result_keypoint = results['xyn'].numpy()[0]

file = open('geek.txt','w')
file.write(result_keypoint)
file.close()"""