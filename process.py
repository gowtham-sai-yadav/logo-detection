import torch
from ultralytics import YOLO
import av
import json
from torchvision import transforms
import numpy as np

# Print YOLOv8 version information
import ultralytics
print(f"YOLOv8 version: {ultralytics.__version__}")
print(f"Torch version: {torch.__version__}")

# Load the YOLOv8 model
model = YOLO('PespiAndCocaCola.pt')

# Define the transform to preprocess the frames
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640)),  # Resize to match YOLOv8 input size
    transforms.Pad((0, 0, 0, 0), fill=0),  # Pad if necessary
])

def extract_frames(video_path):
    container = av.open(video_path)
    frames = []
    timestamps = []
    
    for frame in container.decode(video=0):
        img = frame.to_image()
        frame_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        frames.append(frame_tensor)
        timestamps.append(frame.time)
    
    return frames, timestamps

def calculate_size_and_distance(box, img_width, img_height):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    width = x2 - x1
    height = y2 - y1
    size = (width * height) / (img_width * img_height)  # Relative size
    
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    img_center_x, img_center_y = img_width / 2, img_height / 2
    distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
    relative_distance = distance / np.sqrt(img_width**2 + img_height**2)  # Normalize distance
    
    return size, relative_distance

def detect_logos(frames, timestamps):
    pepsi_detections = []
    cocacola_detections = []
    
    for i, frame in enumerate(frames):
        results = model(frame)
        for result in results:
            boxes = result.boxes
            img_width, img_height = result.orig_shape
            for box in boxes:
                cls = int(box.cls[0])
                size, distance = calculate_size_and_distance(box, img_width, img_height)
                detection = {
                    "timestamp": timestamps[i],
                    "size": size,
                    "distance": distance
                }
                if model.names[cls] == 'Pepsi':
                    pepsi_detections.append(detection)
                elif model.names[cls] == 'CocaCola':
                    cocacola_detections.append(detection)
    
    return pepsi_detections, cocacola_detections

def save_results(pepsi_detections, cocacola_detections, output_file='output.json'):
    results = {
        'Pepsi_detections': pepsi_detections,
        'CocaCola_detections': cocacola_detections
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

# Main function to run the pipeline
def main(video_path):
    print("Extracting frames...")
    frames, timestamps = extract_frames(video_path)
    print("Detecting logos...")
    pepsi_detections, cocacola_detections = detect_logos(frames, timestamps)
    print("Saving results...")
    save_results(pepsi_detections, cocacola_detections)
    print("Done!")

# Example usage:
if __name__ == "__main__":
    video_path = 'demo_video.mp4'
    main(video_path)