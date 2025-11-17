Assignment 4 – Template Matching (Correlation) & Panorama Stitching
Part 1 — Object Detection Using Template Matching (Correlation Method)

This part demonstrates object detection using Template Matching through correlation.
The template images are taken from completely different scenes, not cropped from the test images.

Completed Tasks

Implemented correlation-based template matching

Used 10 different objects for evaluation

Templates come from different scenes

Detection results visualized

Input Images

Below are examples of the test images used:

<img src="Assignment 4/Part-1/images/photo_5827707215912045578_y.jpg" width="300"> <img src="Assignment 4/Part-1/images/photo_5827707215912045579_y.jpg" width="300"> <img src="Assignment 4/Part-1/images/photo_5827707215912045580_y.jpg" width="300"> <img src="Assignment 4/Part-1/images/photo_5827707215912045581_y.jpg" width="300">
Part 2 — Image Stitching (Homography-based Panorama)

This part implements a full image stitching pipeline using:

SIFT feature extraction

Feature matching

RANSAC-based homography estimation

Warping + blending to produce a panorama

Output Panorama

Final stitched panorama:
<img src="Assignment 4/Part-1/output_panorama.png" width="600">

Phone Panorama (Comparison)
<img src="Assignment 4/Part-1/phone_panorama.jpg" width="600">
Repository Structure
Assignment 4/
│
├── Part-1/
│   ├── images/
│   │   ├── photo_5827707215912045578_y.jpg
│   │   ├── photo_5827707215912045579_y.jpg
│   │   ├── photo_5827707215912045580_y.jpg
│   │   └── photo_5827707215912045581_y.jpg
│   ├── output_panorama.png
│   ├── phone_panorama.jpg
│   └── code_files_here.py
│
└── README.md

How to Run
python template_matching.py
python panorama_stitching.py

Summary

This assignment covers:

Template matching via correlation

Multi-object detection using external templates

Full panorama stitching pipeline

Visual comparison against mobile panorama
