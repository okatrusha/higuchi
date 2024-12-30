import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from matplotlib import pyplot as plt


def higuchi_fractal_dimension(image_path, k_max=10):
    """
    Calculate the fractal dimension of an image using Higuchi's method.
    
    Args:
    - image_path (str): Path to the image file.
    - k_max (int): Maximum interval for calculating curve lengths.
    
    Returns:
    - float: Higuchi Fractal Dimension (HFD).
    """
    # Load image and convert to grayscale
    image = Image.open(image_path).convert('L')
    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.Canny(image, 50, 300)  # Extract edges
    plt.imshow(image, cmap="gray")
    plt.show()

    pixel_values = np.array(image)
    
    # Flatten the image into a 1D array
    signal = pixel_values.flatten()
    
    def curve_length(signal, k):
        """
        Compute the curve length for a given interval k.
        """
        L = []
        for m in range(k):
            # Extract m-th subsequence
            subsequence = signal[m::k]
            n = len(subsequence)
            norm = (n - 1) / (k * (n - 1))  # Normalization factor
            diff_sum = np.sum(np.abs(np.diff(subsequence)))
            L.append(norm * diff_sum)
        return np.mean(L)
    
    # Compute lengths for different k values
    lengths = []
    for k in range(1, k_max + 1):
        lengths.append(curve_length(signal, k))
    
    # Log-log fit to determine fractal dimension
    log_k = np.log(range(1, k_max + 1))
    log_lengths = np.log(lengths)
    
    coeffs = np.polyfit(log_k, log_lengths, 1)
    hfd = -coeffs[0]  # Slope gives the Higuchi Fractal Dimension
    
    # Plot log-log graph
    plt.figure(figsize=(6, 6))
    plt.scatter(log_k, log_lengths, label="Data points")
    plt.plot(log_k, np.polyval(coeffs, log_k), color='red', label=f"Fit line (Slope: {hfd:.2f})")
    plt.xlabel("log(k)")
    plt.ylabel("log(L(k))")
    plt.legend()
    plt.show()
    
    return hfd

# Example usage
# image_path = "C:\\Work\\XTF\\sonobot-5077\\SSS Images\\sonobot-5077_20240403_125212_01.png"
image_path = "C:\\Work\\XTF\\sonobot-5077\\SSS Images\\sonobot-5077_20240403_125212_06.png"
# image_path = "C:\\Temp\\romanesco-broccoli-vegetable-close-up-simon-bratt-photography-lrps.jpg"
hfd = higuchi_fractal_dimension(image_path)
print(f"Higuchi Fractal Dimension: {hfd}")