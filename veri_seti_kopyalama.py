import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# ---- Ayarlar ----
INPUT_CSV = r"C:\Users\rezmi\Desktop\Dil_modeli\dataset\labels.csv"
INPUT_IMG_DIR = r"C:\Users\rezmi\Desktop\Dil_modeli\dataset\images"
OUTPUT_IMG_DIR = r"C:\Users\rezmi\Desktop\Dil_modeli\artirilmis_veri\images"
OUTPUT_CSV = r"C:\Users\rezmi\Desktop\Dil_modeli\artirilmis_veri\labels.csv"

lock = Lock()
all_new_records = []
image_counter = 1

# ---- Yardımcı Fonksiyonlar ----
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderValue=255)

def apply_contrast(image, factor=1.2):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def recolor_image(image, bg_color, text_color):
    colored = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        colored[:, :, i] = np.where(image < 127, text_color[i], bg_color[i])
    return colored

def augment_image(image, label):
    global image_counter
    local_records = []

    h, w = image.shape[:2]
    zoom_factors = [1.0, 0.98, 0.95]
    rotation_angles = [0, 10, 15]
    blur_levels = [0, 1]
    color_pairs = [
        ((255, 255, 255), (0, 0, 0)),
        ((255, 255, 255), (0, 128, 0)),
        ((255, 230, 200), (30, 30, 30)),
        ((255, 255, 0), (0, 0, 128)),
        ((240, 240, 240), (20, 20, 20)),
    ]

    for zoom in zoom_factors:
        zh, zw = int(h * zoom), int(w * zoom)
        resized = cv2.resize(image, (zw, zh), interpolation=cv2.INTER_LINEAR)
        canvas = np.ones((h, w), dtype=np.uint8) * 255
        y_offset = (h - zh) // 2
        x_offset = (w - zw) // 2
        canvas[y_offset:y_offset+zh, x_offset:x_offset+zw] = resized

        for angle in rotation_angles:
            rotated = rotate_image(canvas, angle)

            for blur in blur_levels:
                blurred = cv2.GaussianBlur(rotated, (3, 3), 0) if blur else rotated

                for bg_color, text_color in color_pairs:
                    contrasted = apply_contrast(blurred, 1.3)
                    recolored = recolor_image(contrasted, bg_color, text_color)
                    final = apply_contrast(recolored, 1.1)

                    with lock:
                        new_name = f"{image_counter:05d}.png"
                        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, new_name), final)
                        local_records.append((new_name, label))
                        image_counter += 1

    with lock:
        all_new_records.extend(local_records)

# ---- Ana Akış ----
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
df = pd.read_csv(INPUT_CSV, sep=";", encoding="utf-8-sig")

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = []
    for i, row in tqdm(list(df.iterrows()), desc="Augment işlemi devam ediyor..."):
        try:
            img_path = os.path.join(INPUT_IMG_DIR, row['file'])
            label = str(row['text']).strip()
            if os.path.exists(img_path) and label:
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                futures.append(executor.submit(augment_image, image, label))
        except Exception as e:
            print(f"Hata oluştu: {e} (Satır: {i})")

    for f in futures:
        f.result()

# ---- Yeni CSV Kaydet ----
all_new_records.sort(key=lambda x: int(x[0].split(".")[0]))
new_df = pd.DataFrame(all_new_records, columns=["file", "text"])
new_df.to_csv(OUTPUT_CSV, sep=";", index=False, encoding="utf-8-sig")
print("✅ Tüm görseller augment edildi ve CSV doğru sırayla kaydedildi.")
