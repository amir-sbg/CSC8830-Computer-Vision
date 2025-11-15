import cv2
import numpy as np


def load_color_image(path: str) -> np.ndarray:
    """Load RGB image normalized to [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Image not found: " + path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def create_gaussian_kernel(ksize=25, sigma=5.0):
    """Create a 2D Gaussian kernel."""
    g = cv2.getGaussianKernel(ksize, sigma)
    kernel = g @ g.T
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def psf_to_otf(psf, shape):
    H, W = shape
    h, w = psf.shape

    padded = np.zeros((H, W), dtype=np.float32)

    # place kernel so that its center is exactly in the corner
    cy, cx = h // 2, w // 2
    padded[:h-cy, :w-cx] = psf[cy:, cx:]
    padded[:h-cy, W-cx:] = psf[cy:, :cx]
    padded[H-cy:, :w-cx] = psf[:cy, cx:]
    padded[H-cy:, W-cx:] = psf[:cy, :cx]

    return np.fft.fft2(padded)



def wiener_filter(blurred: np.ndarray, kernel: np.ndarray, K=0.002):
    """
    Wiener filtering per channel.
    K is the noise-to-signal ratio; small K = sharper restoration.
    """
    restored = np.zeros_like(blurred)

    for c in range(3):
        channel = blurred[:, :, c]
        G = np.fft.fft2(channel)

        H = psf_to_otf(kernel, channel.shape)
        H_conj = np.conjugate(H)

        # Wiener filter formula
        F = (H_conj / (H * H_conj + K)) * G

        f = np.fft.ifft2(F)
        restored[:, :, c] = np.real(f)

    restored = np.clip(restored, 0, 1)
    return restored


def main():
    # Load original
    img = load_color_image("Book.jpg")

    # Create Gaussian kernel
    kernel = create_gaussian_kernel(ksize=25, sigma=5.0)

    # Blur the RGB image
    blurred = np.zeros_like(img)
    for c in range(3):
        blurred[:, :, c] = cv2.filter2D(img[:, :, c], -1, kernel)

    # Restore image using Wiener filter in Fourier domain
    restored = wiener_filter(blurred, kernel, K=0.001)

    # Save (convert RGB → BGR for cv2.imwrite)
    cv2.imwrite("Book_blurred.png", (blurred[:, :, ::-1] * 255).astype(np.uint8))
    cv2.imwrite("Book_restored.png", (restored[:, :, ::-1] * 255).astype(np.uint8))
    cv2.imwrite("Book_original.png", (img[:, :, ::-1] * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
