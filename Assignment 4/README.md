# Assignment 4 – Template Matching (Correlation) & Panorama Stitching

## Part 1 — Object Detection Using Template Matching (Correlation)

This part demonstrates **object detection using Template Matching through correlation**.  
The template images are taken from **completely different scenes**, not cropped from the test images.

### ✔️ Completed Tasks
- Implemented correlation-based template matching  
- Evaluated **10 different objects**  
- Templates are from different scenes  
- Detection results visualized  

---

## 📥 Input Images

Below are examples of the test images used:

<img src="Part-1/images/photo_5827707215912045578_y.jpg" width="100">
<img src="Part-1/images/photo_5827707215912045579_y.jpg" width="100">
<img src="Part-1/images/photo_5827707215912045580_y.jpg" width="100">
<img src="Part-1/images/photo_5827707215912045581_y.jpg" width="100">

---

## Part 2 — Image Stitching (Homography-based Panorama)

This part implements a full image stitching pipeline using:

- SIFT feature extraction  
- Feature matching  
- RANSAC-based homography estimation  
- Warping + blending to produce a panorama  

### 🖼 Output Panorama  
**Final stitched panorama:**  
<img src="output_panorama.png" width="650">

### 📱 Phone Panorama (Comparison)  
<img src="phone_panorama.jpg" width="650">

---

## 📁 Repository Structure
