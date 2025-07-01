## Overview
This project detects and recognizes food items and tableware on a restaurant table from top-down video footage.  
Object detection is performed with **YOLOv11-small**.

## Dataset
The `data` directory follows the usual split (`train`, `val`, `test`) and includes only a few sample images.

## Training
A Jupyter notebook with the training pipeline is located in `notebooks/`.

## Inference
* `main.py` contains the main logic for running object detection on video.  
* All paths, grouped annotation lists, and other settings are configured in `config.yaml`.