# --- IMPORTS ---
import cv2
import numpy as np
from PIL import Image, ImageOps

# --- STEP 1: Load the Image ---
img_path = "distorted_qr.png"  # replace with your file path
image = Image.open(img_path)

# Convert to grayscale for simplicity
gray = ImageOps.grayscale(image)
img_cv = np.array(gray)

# --- STEP 2: Denoising ---
# Remove small specks/noise while keeping edges
denoised = cv2.medianBlur(img_cv, 3)

# --- STEP 3: Contrast Enhancement ---
# Improves visibility of the black/white QR blocks
enhanced = cv2.convertScaleAbs(denoised, alpha=2.5, beta=0)

# --- STEP 4: Adaptive Thresholding ---
# Binarizes the image (turns it pure black & white)
binary = cv2.adaptiveThreshold(
    enhanced, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 35, 10
)

# --- STEP 5: Morphological Cleaning ---
# Removes noise & smooths block edges
kernel = np.ones((3, 3), np.uint8)
morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

# --- STEP 6: Try to Decode the QR Code ---
detector = cv2.QRCodeDetector()

# Try decoding several filtered versions for robustness
attempts = [img_cv, denoised, enhanced, binary, morph]
for idx, img in enumerate(attempts, 1):
    data, points, _ = detector.detectAndDecode(img)
    if data:
        print(f"✅ QR Decoded at step {idx}:", data)
        break
else:
    print("❌ QR not decoded — try adjusting alpha/beta or block size in thresholding.")
