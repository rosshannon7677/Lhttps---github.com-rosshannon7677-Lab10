import cv2
from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "cars1.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties for the output file
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 format
out = cv2.VideoWriter('output_cars1.mp4', fourcc, fps, (frame_width, frame_height))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Write the frame to the output video file
        out.write(annotated_frame)

        # Display the annotated frame (optional)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and video write objects, and close all OpenCV windows
cap.release()
out.release()
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
