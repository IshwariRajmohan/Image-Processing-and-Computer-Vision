import cv2
import numpy as np
import os

def preprocess_image(image, filename):

    base_path = f"output/{filename}/preprocessing"
    os.makedirs(base_path, exist_ok=True)

    resized = cv2.resize(image, (500, 500))
    cv2.imwrite(f"{base_path}/1_resized.jpg", resized)

    h, w = resized.shape[:2]
    cropped = resized[h//10:h-h//10, w//10:w-w//10]
    cv2.imwrite(f"{base_path}/2_cropped.jpg", cropped)

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{base_path}/3_gray.jpg", gray)

    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    cv2.imwrite(f"{base_path}/4_blur.jpg", blurred)

    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"{base_path}/5_threshold.jpg", thresh)

    edges = cv2.Canny(blurred, 50, 150)
    cv2.imwrite(f"{base_path}/6_edges.jpg", edges)

    norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(f"{base_path}/7_normalized.jpg", norm)

    hist_eq = cv2.equalizeHist(blurred)
    cv2.imwrite(f"{base_path}/8_hist_eq.jpg", hist_eq)

    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(f"{base_path}/9_morph.jpg", morph)

    denoise = cv2.medianBlur(hist_eq, 5)
    cv2.imwrite(f"{base_path}/10_denoise.jpg", denoise)

    final = cv2.cvtColor(denoise, cv2.COLOR_GRAY2BGR)

    return final