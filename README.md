
---

# Crosswalk and Motion Detection using YOLOv8

This repository provides tools and resources for detecting crosswalks and motion using the YOLOv8 model. It has been fine-tuned to deliver better results specifically for crosswalk detection.

## Repository Structure

- `crosswalk_model/`: Contains the fine-tuned YOLOv8 model for crosswalk detection.
- `dataset/video/`: Place your videos here for motion detection.
- `video_results/`: After processing, the resulting videos with detected motions and crosswalks are stored here.

## Sample Results

To give you an idea of the output, here's a sample result from our processed videos:

Sample Video Result: https://drive.google.com/file/d/1Bt_TWh5c-upXJQq6UUFrIHicw-3bwtp7/view?usp=sharing


## Getting Started

1. **Crosswalk Detection Model**: For a deeper understanding and details about the dataset used for fine-tuning the YOLOv8 model, please refer to this repository: [Crosswalks Detection using YOLO](https://github.com/xN1ckuz/Crosswalks-Detection-using-YOLO).

2. **Motion Detection**:
   - Place the video you wish to analyze in the `dataset/video` folder.
   - Update the `motion_detection.py` script with the exact location of your video.
   - Run the `motion_detection.py` script to process the video.
   - Check the `video_results` folder for the processed video.

---
