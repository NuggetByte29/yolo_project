# Automated Lime Ripeness Inspection System

This repository contains a computer vision application designed to detect and classify lime fruits in real-time. By utilizing YOLOv11, the system differentiates between **ripe** and **unripe** limes, serving as a functional prototype for automated visual inspection in agricultural or food processing environments.

## Features
* **Real-Time Classification:** Captures live webcam feeds and processes frames to identify the ripeness state of limes instantly.
* **Custom Object Detection:** Built on the YOLO architecture, trained specifically on a custom dataset of ripe and unripe lime imagery.
* **Automated Data Preparation:** Includes utility scripts for seamless training and validation dataset splitting.

## Repository Structure
* `run_dashboard.py`: The main execution script. It initializes the camera stream, loads the custom-trained weights (`best.pt`), and runs the live inference.
* `train_val_split.py`: A data processing script used to automatically partition image datasets into proper training and validation folders.
* `check_gpu.py`: A quick diagnostic tool to verify CUDA and GPU availability to ensure hardware-accelerated performance.
* `data.yaml`: The configuration file defining the dataset structure and target classes (ripe, unripe).
* `yolo11n.pt`: The base nano model weights used as the starting point for training.

## How to Run
1. Ensure your Python environment has the necessary dependencies installed (such as `ultralytics` and `opencv-python`).
2. Run the diagnostic script to confirm your system is utilizing the GPU for optimal frame rates:
   ```bash
   python check_gpu.py