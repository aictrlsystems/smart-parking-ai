# smart-parking-ai

This project implements a hybrid deep learning and rule-based AI system for parking lot monitoring. It uses the YOLOv8n object detection model to detect vehicles and pedestrians from video footage, and applies logical rules to generate alerts based on occupancy and traffic conditions.

The system is designed to run on edge AI hardware (e.g., NVIDIA Jetson Orin Nano) to support real-time edge operation without relying on cloud-based computation.

## Relationship to the Book Version

This repository provides a public version of the Smart Parking AI implementation described in Chapter 9 of *Practical AI Control Methods*. The code follows the same core workflow presented in the book: YOLOv8n-based object detection, OpenCV video processing, rule-based alert generation, bounding-box annotation, and annotated video output.

This public version includes minor implementation updates for repository use, including command-line video input, configurable output directory selection, optional display mode, and additional runtime checks. These changes do not alter the main purpose of the project, which is to demonstrate a hybrid deep learning and rule-based AI system for parking lot monitoring.

## Features

- Frame-by-frame object detection using YOLOv8n
- Rule-based alerts based on car, pedestrian, and motorcycle counts
- Bounding box overlays and alert banners on video output
- Video frame processing using OpenCV
- Tested on NVIDIA Jetson Orin Nano with JetPack 6.1

## Requirements

This project can run on a regular laptop/desktop or on NVIDIA Jetson hardware.

For a regular laptop/desktop environment:

- Python 3.10 or newer
- ultralytics==8.3.78
- opencv-python
- numpy

  Install the standard Python dependencies from the project folder using:

  ```bash
  pip install -r requirements.txt
  ```

For NVIDIA Jetson Orin Nano deployment:

- Python 3.10
- Jetson-compatible PyTorch build
- Torchvision compiled for Jetson
- ultralytics==8.3.78
- opencv-python
- numpy
- onnxruntime-gpu==1.20.0, if using the broader Jetson vision-stack setup

## Rule-Based Alerts

- Alerts if the number of vehicles exceeds 60
- Alerts if more than 5 pedestrians are detected
- Alerts if more than 3 motorcycles are present
- Critical alert if both vehicles > 8 and people > 5

Note: In this implementation, cars, trucks, and buses are grouped together as vehicles for the rule-based counts. These thresholds are for demonstration purposes and can be adjusted based on the specific needs or configuration of the parking lot.

## Example Output

Annotated video frame showing bounding boxes and real-time alert banner.

## Input Video

This repository does not include video footage. Users must provide their own parking lot video file.

Place the video file in the same folder as `smart2_ai_video.py`, then open a terminal or command prompt in that folder and run:

```bash
python smart2_ai_video.py --video mall_lot_video.mp4 --output-dir yolo_output
```

If the video has a different filename, replace `mall_lot_video.mp4` with the exact filename.

Example:

```bash
python smart2_ai_video.py --video your_video_name.mp4 --output-dir yolo_output
```

To display the annotated video while processing, add `--display`:

```bash
python smart2_ai_video.py --video your_video_name.mp4 --output-dir yolo_output --display
```

The script processes the video frame by frame and saves the annotated output video here:

```text
yolo_output/processed_parking_lot_video.mp4
```

If `--display` is used, pressing `q` stops processing early. In that case, only the frames processed before stopping are saved.

## Open Source License

This project is released as open-source software under the GNU Affero General Public License v3.0 (AGPL-3.0).

You may use, study, modify, and redistribute this software under the terms of the AGPL-3.0 license.

See the `LICENSE` file for the full license text.

This software is provided without warranty of any kind. The author makes no guarantees regarding accuracy, reliability, safety, or fitness for any particular use.

## Citation

If you use this project, please cite:

- R. Pasko, Jr., *Practical AI Control Methods: Neural, Fuzzy, and Reinforcement Learning Approaches*. CRC Press, 2027.

This project supports Chapter 9 of the book, which presents a hybrid deep-learning and rule-based AI system for parking lot monitoring, occupancy assessment, and alert generation.
