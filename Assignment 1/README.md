# 📐 CSC 8830 – Homework 1  
### Real-World Dimension Estimation Using Perspective Projection  
**Author:** Amir Sabbaghziarani  
**Course:** CSc 8830 – Computer Vision  
**Instructor:** Dr. Ashwin Ashok  
**Semester:** Spring 2026  

---

## 📝 Overview  
This homework implements two tasks:

- **(1) Python script for real-world measurement** using camera intrinsics and perspective projection equations.  
  The script allows a user to click two points on an image of an object taken at a known distance from the camera, and computes the real-world distance between those points.

- **(2) A browser-based web application** that performs the same measurement task using an uploaded image.  
  The user clicks two points on the displayed image, and the web page computes the real-world dimension in centimeters.

Both implementations rely on the same principles of perspective projection, intrinsic matrix scaling, ray back-projection, and Euclidean distance computation.

---

## 🎯 Objectives

### **1. Python Measurement Script**
- Loads an image containing a known calibration target (e.g., a chessboard pattern).  
- Rescales the intrinsic matrix to match the image resolution.  
- Uses two user-selected pixel coordinates to generate 3D rays from the camera.  
- Multiplies rays by a user-specified camera–object distance \( Z_c \).  
- Computes the real-world Euclidean distance between the two resulting 3D points.  
- Displays and saves the resulting visualization and measurement.

**Key Features**
- Click-based point selection  
- Automatic intrinsic matrix re-scaling  
- Perspective back-projection  
- Visual overlay of selected points  
- Final export as a figure  

Sample output images are provided under the `/Output/` folder.

---

### **2. Web-Based Real-World Measurement App**
- Runs directly in any modern browser (Chrome, Firefox, Safari).  
- Upload any image containing an object.  
- Click two points representing the measurement endpoints.  
- The app computes and displays:  
  - Pixel distance  
  - Real-world estimated length  

The application is OS-agnostic and uses only standard JavaScript + HTML.  
User interaction assumptions:  
- User manually clicks the two measurement points.  
- The user knows or inputs the camera–object distance.  

---

## 📁 Output Files  
The following example outputs from the Python script and web app are included in this repository:

- **/Output/HW1-1.png**  
  ![HW1-1](/Output/HW1-1.png)

- **/Output/HW1-2.png**  
  ![HW1-2](/Output/HW1-2.png)

---

## 🧠 Method Summary

### **Perspective Projection Approach**
Given intrinsic matrix \( K \) and pixel coordinate \( u \), the back-projected 3D point is:

\[
X = Z_c \, K^{-1} u
\]

For two selected points \( u_1, u_2 \):

\[
X_1 = Z_c \, K^{-1} u_1, \quad X_2 = Z_c \, K^{-1} u_2
\]

The real-world distance estimate:

\[
d = \lVert X_1 - X_2 \rVert_2
\]

---

## ▶️ How to Run the Python Script

1. Place an input image (e.g., `ChessBoard.jpeg`) in the working directory.  
2. Run:

```bash
python measure_length.py
