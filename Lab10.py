from ultralytics import YOLO

# Load a YOLOv8 model. You can choose from different types of models like 'yolov8n.pt', 'yolov8n-seg.pt', 'yolov8n-pose.pt',
# or use a custom trained model by specifying its path.
model = YOLO('yolov8n.pt')  # This is an example of loading an official Detect model

# Perform tracking with the model. You can specify the source as a video file path or a URL.
# The 'show=True' argument will display the output.
results = model.track(source="C:\Lab10\\traffic.mp4", show=True)  # Tracking with default tracker

# Optionally, you can use a different tracker like ByteTrack by specifying the 'tracker' argument.
# results = model.track(source="path_to_your_video_file_or_URL", show=True, tracker="bytetrack.yaml")  # Tracking with ByteTrack tracker
