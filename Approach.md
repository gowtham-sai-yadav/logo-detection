# Approach Document: Logo Detection in Videos

## 1. Introduction

This document outlines the approach taken to develop a machine learning pipeline for detecting Pepsi and Coca-Cola logos in video files. The primary objective was to create a system that can process video input, identify logos in each frame, and output the detection results in a specified JSON format.

## 2. Technology Stack

### 2.1 Python
Python was chosen as the primary programming language due to its extensive libraries for machine learning and video processing, as well as its ease of use and readability.

### 2.2 YOLOv8
We selected YOLOv8 (You Only Look Once version 8) as our object detection model for several reasons:
- State-of-the-art performance in object detection tasks
- Fast inference speed, suitable for video processing
- Ability to detect multiple objects in a single pass
- Extensive pre-trained models and easy fine-tuning capabilities

### 2.3 PyTorch
PyTorch serves as the underlying deep learning framework, chosen for its dynamic computation graphs and extensive community support.

### 2.4 av Library
The `av` library was used for video frame extraction due to its efficiency and compatibility with various video formats.

## 3. Implementation Details

### 3.1 Video Processing
- The `av` library is used to extract frames and their corresponding timestamps from the input video.
- Each frame is preprocessed (resized and normalized) to match the input requirements of the YOLOv8 model.

### 3.2 Logo Detection
- The YOLOv8 model, pre-trained on a custom dataset of Pepsi and Coca-Cola logos, processes each frame.
- The model outputs bounding boxes, confidence scores, and class predictions for each detected logo.

### 3.3 Post-processing
- We calculate the size of each detected logo relative to the frame size.
- The distance of each logo from the center of the frame is computed and normalized.
- Results are aggregated and formatted according to the specified JSON structure.

## 4. Challenges and Solutions

### 4.1 Model Training
Challenge: Obtaining a diverse dataset of Pepsi and Coca-Cola logos in various contexts.
Solution: We augmented publicly available datasets with custom-collected images to improve model robustness.

### 4.2 False Positives
Challenge: Initial tests showed false positives, especially with similar-looking objects.
Solution: We fine-tuned the model with hard negative mining, focusing on common false positive cases.
    
### 4.3 Processing Speed
Challenge: Initial implementation was too slow for real-time processing.
Solution: We optimized the frame extraction process and utilized GPU acceleration where available.

## 5. Assumptions

- The input video is assumed to be in a format supported by the `av` library.
- The model assumes consistent lighting conditions; extreme variations may affect performance.
- We assume that logos occupy a significant portion of the frame when present; very small logos may be missed.

## 6. Potential Improvements

1. Multi-scale detection: Implement a multi-scale approach to better detect logos of varying sizes.
2. Temporal consistency: Utilize information from adjacent frames to improve detection consistency.
3. Model quantization: Explore model quantization techniques to improve inference speed on less powerful hardware.
4. Active learning pipeline: Develop a system for continuous model improvement based on user feedback.

## 7. Conclusion

The current implementation provides a robust solution for detecting Pepsi and Coca-Cola logos in video content. While there's room for improvement, the system meets the specified requirements and provides a solid foundation for further enhancements.