# Augmented Reality Video Overlay

This project demonstrates an augmented reality (AR) application using OpenCV and Python. The goal is to overlay a video onto a detected target image within a live webcam feed. By leveraging feature matching with the ORB algorithm and computing a homography, the video is accurately warped to fit the target's perspective in real-time.

## Features

- **Real-Time Webcam Capture:** Streams live video from your default webcam.
- **Target Detection:** Uses ORB feature detection to identify a specified target image within the webcam feed.
- **Homography Computation:** Computes a transformation matrix (homography) with RANSAC to map the target image onto the webcam view.
- **Video Warping & Overlay:** Warps a video frame to the detected target area and overlays it onto the live feed.
- **Debug Visualization:** Stacks multiple processed images (e.g., original webcam feed, target image, feature matches, warped video, augmented output) for easier debugging and visualization.

## Requirements

- Python 3.x
- [OpenCV](https://opencv.org/) (`opencv-python`)
- [NumPy](https://numpy.org/)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name

