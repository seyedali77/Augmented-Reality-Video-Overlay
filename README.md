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
## Setup
1. **Media Files:**

   Place your target image (e.g., ImageTarget.jpg) in the project directory.
   Place the video file (e.g., video.mp4) you wish to overlay in the project directory.
2. **Script File:**

   Ensure that the main Python script (e.g., main.py) contains the provided code that utilizes the target image and video.


## How It Works
**ORB Feature Detection:**
   The script initializes an ORB detector to compute keypoints and descriptors for both the target image and the live webcam    feed.

**Feature Matching:**
   A brute-force matcher with k-nearest neighbors (k=2) is used to find good matches between features from the target image     and the webcam frame. A ratio test is applied to filter out weak matches.

**Homography Estimation:**
   When sufficient matches are found, a homography matrix is computed using RANSAC. This matrix maps the target image's         plane onto the corresponding area in the webcam frame.

**Video Warping & Overlay:**
   The video frame is warped using the computed homography matrix so that it fits the perspective of the detected target        area. Bitwise operations merge the warped video with the live webcam feed, creating the AR effect.

**Visualization:**
   For debugging and demonstration, multiple images (original webcam feed, target image, video frame, feature matches,          warped video, augmented output) are stacked into a single display window.
   
