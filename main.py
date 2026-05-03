import cv2
import os
from preprocessing import preprocess_image
from utils import detect_flood

input_folder = "dataset/flood/"

for file in os.listdir(input_folder):

    path = os.path.join(input_folder, file)
    image = cv2.imread(path)

    if image is None:
        continue

    filename = file.split('.')[0]

    print("\n📂 Processing:", file)

    # ---------------- PREPROCESSING ----------------
    preprocessed = preprocess_image(image, filename)

    # ---------------- FLOOD DETECTION ----------------
    result, mask, percentage, level = detect_flood(image)

    print(f"🌊 Flood Area: {percentage:.2f}%")
    print(f"⚠️ Flood Level: {level}")

    # ---------------- SAVE RESULTS ----------------
    result_path = f"output/{filename}/results"
    os.makedirs(result_path, exist_ok=True)

    cv2.imwrite(f"{result_path}/result.jpg", result)
    cv2.imwrite(f"{result_path}/mask.jpg", mask)

    # ---------------- DISPLAY ----------------
    cv2.imshow("Original Image", image)
    cv2.imshow("Preprocessed Image", preprocessed)
    cv2.imshow("Flood Mask", mask)
    cv2.imshow("Final Result", result)

    # ---------------- SIMPLE CONTROL ----------------
    key = cv2.waitKey(0)

    if key == 27:   # ESC key
        print("🛑 ESC pressed. Exiting...")
        break

# Close all windows
cv2.destroyAllWindows()