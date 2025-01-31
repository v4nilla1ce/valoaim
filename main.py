import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (Pretrained, can fine-tune later for Valorant)
crosshair_model = YOLO("yolov8n.pt")  # Replace with custom trained model for better results
enemy_model = YOLO("yolov8n.pt")  # Ideally, train a custom model for Valorant

# Load Video
video_path = "gameplay.mp4"  # Change to your video file
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video
    
    # Convert frame to YOLO format
    results_crosshair = crosshair_model(frame)
    results_enemy = enemy_model(frame)
    
    # Draw crosshair detections
    for result in results_crosshair:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green Box for Crosshair
            
    # Draw enemy detections
    for result in results_enemy:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red Box for Enemies
            
    # Show frame
    cv2.imshow("Valorant Aim Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
