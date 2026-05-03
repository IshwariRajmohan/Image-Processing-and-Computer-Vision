import cv2
import numpy as np

def detect_flood(image):

    # Resize
    image = cv2.resize(image, (500, 500))

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ---------------------------
    # 1. Muddy water detection

    lower1 = np.array([5, 50, 50])
    upper1 = np.array([30, 255, 255])

    lower2 = np.array([0, 0, 0])
    upper2 = np.array([180, 255, 80])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask_hsv = cv2.bitwise_or(mask1, mask2)

    # ---------------------------
    # 2. Remove green (trees)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_hsv = cv2.bitwise_and(mask_hsv, cv2.bitwise_not(green_mask))

    # ---------------------------
    # 3. Morphological cleaning

    kernel = np.ones((7,7), np.uint8)
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel)
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)

    # ---------------------------
    # 4. Remove small regions

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_hsv)

    min_area = 800  # 🔥 tune this if needed

    clean_mask = np.zeros_like(mask_hsv)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            clean_mask[labels == i] = 255

    mask_hsv = clean_mask

    # ---------------------------
    # 5. Remove upper region (buildings)

    h, w = mask_hsv.shape
    mask_hsv[:h//3, :] = 0

    # ---------------------------
    # 6. Safety check

    if np.sum(mask_hsv) == 0:
        print("⚠️ No flood detected")
        return image, mask_hsv, 0, "No Flood"

    # ---------------------------
    # 7. GrabCut refinement

    mask = np.zeros(image.shape[:2], np.uint8)
    mask[mask_hsv > 0] = cv2.GC_PR_FGD
    mask[mask_hsv == 0] = cv2.GC_PR_BGD

    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    try:
        cv2.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

        final_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255, 0
        ).astype('uint8')

    except:
        print("⚠️ GrabCut failed - using HSV mask")
        final_mask = mask_hsv

    # ---------------------------
    # 8. Flood percentage

    total = final_mask.size
    flood = np.sum(final_mask == 255)
    percentage = (flood / total) * 100

    # ---------------------------
    # 9. Flood level

    if percentage < 10:
        level = "Low Flood"
    elif percentage < 30:
        level = "Medium Flood"
    else:
        level = "High Flood"

    # ---------------------------
    # 10. Overlay result

    result = image.copy()
    result[final_mask == 255] = [0, 0, 255]

    # Add text
    cv2.putText(result, f"{level} ({percentage:.2f}%)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    return result, final_mask, percentage, level