import cv2
import numpy as np

def detect_objects(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of a color to detect (example: blue color)
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detected_objects = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Optionally, filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            detected_objects.append((x, y, w, h))

    return detected_objects

# Load your video
cap = cv2.VideoCapture("C:\\Users\\rossh\\Lab10\\traffic3.mp4")

# Loop through each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    detected_objects = detect_objects(frame)

    # Placeholder for object tracking - define this function based on your tracking needs
    # tracked_objects = track_objects(detected_objects)

    # Drawing detected objects on the frame
    for (x, y, w, h) in detected_objects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all frames
cap.release()
cv2.destroyAllWindows()





#rom ultralytics import YOLO

# Load a YOLOv8 model. You can choose from different types of models like 'yolov8n.pt', 'yolov8n-seg.pt', 'yolov8n-pose.pt',
# or use a custom trained model by specifying its path.
#model = YOLO('yolov8n.pt')  # This is an example of loading an official Detect model

# Perform tracking with the model. You can specify the source as a video file path or a URL.
# The 'show=True' argument will display the output.
#results = model.track(source="C:\Lab10\\traffic.mp4", show=True)  # Tracking with default tracker

# Optionally, you can use a different tracker like ByteTrack by specifying the 'tracker' argument.
# results = model.track(source="path_to_your_video_file_or_URL", show=True, tracker="bytetrack.yaml")  # Tracking with ByteTrack tracker
