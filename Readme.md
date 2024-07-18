# Logo Detection in Videos

This project implements a machine learning pipeline to detect Pepsi and Coca-Cola logos in video files using YOLOv8. The system processes video input, identifies logos in each frame, and outputs the detection results in a JSON format.

## Features

- Detects Pepsi and Coca-Cola logos in video files
- Uses YOLOv8 for accurate object detection
- Extracts video frames using the `av` library
- Outputs detection results including timestamps, logo sizes, and distances from frame center
- Supports various video formats (e.g., .mp4, .avi, .mov)

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- av
- numpy
- torchvision

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/gowtham-sai-yadav/logo-detection.git
   cd logo-detection
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the YOLOv8 model file `PespiAndCocaCola.pt` and place it in the project root directory. 
   (Note: You may need to train this model or obtain it from a specific source. If so, include instructions here.)

## Usage

To run the logo detection pipeline:

```
python process.py path/to/your/video.mp4
```

Replace `path/to/your/video.mp4` with the path to the video file you want to analyze.

## Output

The script generates an `output.json` file in the project directory with the following structure:

```json
{
    "Pepsi_detections": [
        {
            "timestamp": 10.1,
            "size": 0.15,
            "distance": 0.3
        },
        ...
    ],
    "CocaCola_detections": [
        {
            "timestamp": 20.3,
            "size": 0.2,
            "distance": 0.1
        },
        ...
    ]
}
```

- `timestamp`: The time in seconds when the logo was detected
- `size`: The relative size of the logo in the frame (0 to 1)
- `distance`: The relative distance of the logo from the center of the frame (0 to 1)

## demo
watch this loom.com video to know how it works {[loom_video](https://www.loom.com/share/3babf4517a8749609e5f474f578269aa?sid=468152f8-ec71-4e8d-b7b9-a39351371822)}

## Approach

For a detailed explanation of the approach used in this project, including design decisions and potential improvements, please refer to the `APPROACH.md` file in this repository.

or refer to this link {https://docs.google.com/document/d/1on_U4pKK0PJOpnYT7oyJhm5ajBOXfNvwa49FeoRs8Vg/edit?usp=sharing}

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- PyAV developers for the `av` library

## Contact

If you have any questions or feedback, please open an issue in this repository or contact [gowthamyadav022@gmail.com].