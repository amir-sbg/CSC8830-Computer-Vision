# 📸 Assignment – Image Stitching (Panorama Construction)

This repository implements **image stitching using feature-based homography** and compares the result with a **mobile phone panorama**.  
A minimum of **4 landscape** or **8 portrait** images must be used.  
Below are the images used, the final results, and the full explanation of the process.

---

# Part 1 — Input Images (Captured Manually)

The following images were captured using a regular camera in **landscape mode** and serve as inputs to the stitching pipeline.

### 📥 Input Frames
<img src="Part-1/images/photo_5827707215912045578_y.jpg" width="160">
<img src="Part-1/images/photo_5827707215912045579_y.jpg" width="160">
<img src="Part-1/images/photo_5827707215912045580_y.jpg" width="160">
<img src="Part-1/images/photo_5827707215912045581_y.jpg" width="160">

These four frames provide enough overlap to compute feature correspondences and estimate homographies.

---

# Part 2 — Image Stitching (Homography-Based Panorama)

This part implements a **complete panorama stitching pipeline** consisting of:

- SIFT feature detection  
- Descriptor extraction  
- Feature matching  
- RANSAC to estimate homography  
- Warping & blending to create a seamless panorama  

### 🧵 Stitching Overview  
The algorithm follows these steps:

1. **Detect features** in all images  
2. **Match keypoints** between consecutive images  
3. **Estimate homography** using RANSAC for robust alignment  
4. **Warp each image** into a common reference frame  
5. **Blend images smoothly** to produce a final panoramic view  

---

### 🖼 Final Output — Stitched Panorama

Below is the panorama produced by the implemented Python stitching pipeline:

<img src="Part-1/output_panorama.png" width="700">

The stitched image successfully aligns overlapping regions and reconstructs a wide field of view.

---

### 📱 Phone Panorama — Comparison

For comparison, the mobile device’s built-in panorama feature produced the following result:

<img src="Part-1/phone_panorama.jpg" width="700">

The phone’s stitching tends to apply stronger blending, exposure correction, and seam smoothing.  
The Python implementation focuses on algorithm correctness and manual feature-based alignment.

---

# 📁 Repository Structure

