# CSC8830-Computer-Vision Homework 3

# Homework 3 – Part 1  
## Gradient Magnitude, Gradient Angle, and Laplacian of Gaussian (LoG)

### 📌 Task Description
In this part of the assignment, we compute three different image derivatives for every image in the dataset:

- **Gradient Magnitude** using Sobel filters  
- **Gradient Angle** using Sobel filter directions (visualized in degrees)  
- **Laplacian of Gaussian (LoG)** obtained by applying Gaussian smoothing followed by a Laplacian operator  

These outputs highlight edge strength, edge orientation, and second-order intensity changes. Each result is saved as a separate image for visualization.

---

### 📁 Input Example  
A sample input image from the dataset:

<img src="part1/hw3_dataset/photo_5839040925936585610_y.jpg" width="300"/>


---

### 📤 Output Demo  
Below are the generated outputs for the above sample image:

#### 🟦 Gradient Angle  
`part1/output_task1/photo_5839040925936585610_y_grad_ang.png`  
<img src="part1/output_task1/photo_5839040925936585610_y_grad_ang.png" width="300"/>

#### 🟩 Gradient Magnitude  
`part1/output_task1/photo_5839040925936585610_y_grad_mag.png`  
<img src="part1/output_task1/photo_5839040925936585610_y_grad_mag.png" width="300"/>

#### 🟥 Laplacian of Gaussian (LoG)  
`part1/output_task1/photo_5839040925936585610_y_log.png`  
<img src="part1/output_task1/photo_5839040925936585610_y_log.png" width="300"/>

---

### 🧠 Summary of Method
- Convert image → grayscale  
- Compute Sobel gradients `gx` and `gy`  
- Compute:
  - `magnitude = sqrt(gx² + gy²)`
  - `angle = arctan2(gy, gx)` → mapped to [0°,180°] → normalized  
- Apply Gaussian smoothing  
- Apply Laplacian filter on the smoothed image  
- Normalize all results to 0–255 for visualization  
- Save outputs as PNG files in `output_task1/`

---

### 📂 Output Directory  
All results are saved under:



---

### ✔️ Notes
- The code automatically processes **all images** in the dataset directory.  
- All visualizations are normalized for easy display.  
- Only results (not code) are shown in this README.

---


