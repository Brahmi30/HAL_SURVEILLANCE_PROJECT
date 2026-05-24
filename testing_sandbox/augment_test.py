import cv2
import os
import numpy as np

# Automatically find the path where this specific test script is saved
SANDBOX_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_IMAGE_PATH = os.path.join(SANDBOX_DIR, "base.jpg")

# 1. Verification Check
if not os.path.exists(BASE_IMAGE_PATH):
    print(f"❌ Error: Cannot find 'base.jpg' inside the testing folder!")
    print(f"Please drop a face image into: {SANDBOX_DIR} and rename it to 'base.jpg'")
    exit()

print("🎯 Base image verified. Initializing augmentation sandbox routines...")
img = cv2.imread(BASE_IMAGE_PATH)

# ==========================================
# 2. DEFINING EXPERIMENTAL VISUAL FILTERS
# ==========================================

def adjust_brightness(image, value):
    """Brightens or darkens the image matrix smoothly using HSV color space."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Safely prevent overflow/underflow when modifying pixel brightness values
    v = np.clip(v.astype(int) + value, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

def tilt_angle(image, angle):
    """Rotates the image to simulate slight head tilting positions."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

def apply_blur(image, intensity=5):
    """Adds a smooth Gaussian blur to simulate webcam streaming motion noise."""
    return cv2.GaussianBlur(image, (intensity, intensity), 0)

# ==========================================
# 3. GENERATION REGISTRY
# ==========================================
# This dictionary maps clear labels to the processed image matrices
test_variations = {
    "test_bright": adjust_brightness(img, 45),
    "test_dim": adjust_brightness(img, -45),
    "test_high_contrast": cv2.convertScaleAbs(img, alpha=1.3, beta=0),
    "test_low_contrast": cv2.convertScaleAbs(img, alpha=0.7, beta=0),
    "test_tilt_left": tilt_angle(img, -8),
    "test_tilt_right": tilt_angle(img, 8),
    "test_mirrored": cv2.flip(img, 1),
    "test_motion_blur": apply_blur(img, 5)
}

# ==========================================
# 4. EXPORT ENGINE
# ==========================================
print("\n🚀 Exporting generated visual assets to sandbox...")
generated_count = 0

for file_label, processed_matrix in test_variations.items():
    output_filename = f"{file_label}.jpg"
    destination_path = os.path.join(SANDBOX_DIR, output_filename)
    
    # Save the output file to disk
    cv2.imwrite(destination_path, processed_matrix)
    generated_count += 1
    print(f" 💾 Saved: {output_filename}")

print(f"\n✅ Pipeline test complete! {generated_count} experimental variations created safely.")
print(f"📂 Open your '{os.path.basename(SANDBOX_DIR)}' folder to view the results side by side!")