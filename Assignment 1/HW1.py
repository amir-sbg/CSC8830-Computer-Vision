import numpy as np
import matplotlib.pyplot as plt

# Intrinsic calibration matrix for the reference image resolution
K_CALIB = np.array([
    [1434.41, 0.00, 949.77],
    [0.00, 1430.68, 541.41],
    [0.00, 0.00, 1.00]
], dtype=float)

# Calibration image size (W, H)
CALIB_SIZE = (1920, 1080)

# Global variables for mouse click handling
click_count = 0
uv1, uv2 = None, None

def rescale_K(K_calib: np.ndarray, calib_size: tuple[int,int], current_size: tuple[int,int]) -> np.ndarray:
    """
    Rescale the intrinsic matrix K from the calibration resolution to the
    resolution of the currently loaded image.

    K_calib: original intrinsic matrix  
    calib_size: (W0, H0) calibration image size  
    current_size: (W, H) size of the runtime image  
    """
    W0, H0 = calib_size
    W, H = current_size

    # Scaling factors for width and height
    sx, sy = W / W0, H / H0

    # Apply scaling to fx, fy, cx, cy
    K = K_calib.copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K

def onclick(event):
    """
    Matplotlib click callback.
    Records the pixel coordinates of two mouse clicks.
    After the second click, the window closes.
    """
    global uv1, uv2, click_count

    # Ignore clicks outside axes
    if event.xdata is None or event.ydata is None:
        return

    # Create homogeneous pixel coordinate
    uv = np.array([[event.xdata], [event.ydata], [1.0]], dtype=float)

    # Save first and second click
    if click_count == 0:
        uv1 = uv
    else:
        uv2 = uv
        plt.close()

    click_count += 1

def main():
    global uv1, uv2, click_count

    # Load input image
    img_path = "ChessBoard.jpeg"
    image = plt.imread(img_path)
    H, W = image.shape[:2]

    # Rescale intrinsic matrix to image size
    K = rescale_K(K_CALIB, CALIB_SIZE, (W, H))
    K_inv = np.linalg.inv(K)

    # Get measurement unit and object depth from user
    unit = (input("Distance unit (cm or mm): ").strip() or "cm")
    zc = float(input(f"Distance from camera Zc (in {unit}): ").strip())

    # Reset click info
    click_count = 0
    uv1 = uv2 = None

    # Display image for selecting two points
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Select two points on the object")
    ax.axis("off")
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    # Ensure two valid points were selected
    if uv1 is None or uv2 is None:
        print("You must click two points inside the image. Try again.")
        return

    # Visualize selected points
    plt.figure()
    plt.imshow(image)
    plt.axis("off")
    plt.scatter(uv1[0,0], uv1[1,0], color='red', s=100, zorder=2)
    plt.scatter(uv2[0,0], uv2[1,0], color='yellow', s=100, zorder=2)
    plt.plot([uv1[0,0], uv2[0,0]], [uv1[1,0], uv2[1,0]], color='green', linewidth=2, zorder=1)

    # Pixel distance between selected points
    pix_dist = float(np.hypot(uv2[0,0] - uv1[0,0], uv2[1,0] - uv1[1,0]))

    # Convert pixel rays to 3D points assuming depth Zc
    xyz1 = (K_inv @ uv1) * zc
    xyz2 = (K_inv @ uv2) * zc

    # Euclidean distance between 3D points
    length = float(np.linalg.norm(xyz1 - xyz2))

    # Show and save result
    title = f"estimated length: {length:.2f} {unit}   (pixel dist: {pix_dist:.1f})"
    plt.title(title)
    plt.savefig("measure_depth_result.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(title)
    print("Saved figure to measure_depth_result.png")

if __name__ == '__main__':
    main()
