import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# --- Dosya Yolları ---
model_path = r"C:\Users\rezmi\Desktop\Dil_modeli\osmanli_model_128.h5"
encoder_path = r"C:\Users\rezmi\Desktop\Dil_modeli\label_encoder_128.pkl"
test_image_path = r"C:\Users\rezmi\Desktop\Dil_modeli\test.png"

# --- Model ve LabelEncoder Yükle ---
model = load_model(model_path)
with open(encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# --- Görseli Gri Tonlamada Yükle ---
img_gray = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

# --- Eşikleme (ters binary + Otsu) ---
_, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# --- Harfleri Bul ---
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- Soldan Sağa Sıralama ---
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

# --- Tahminler ---
tahminler = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Gürültüyü ele: çok küçük şekiller atla
    if w > 10 and h > 10:
        roi = img_gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (128, 128))  # Eğitimdeki gibi
        roi_normalized = roi_resized / 255.0
        roi_reshaped = roi_normalized.reshape(1, 128, 128, 1)

        prediction = model.predict(roi_reshaped, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        tahminler.append(predicted_label)

# --- Sonuç ---
print("🔠 Algılanan karakter sayısı:", len(tahminler))
print("🧠 Model Tahmini:", "".join(tahminler))

# --- Görsel Göster ---
plt.imshow(img_gray, cmap='gray')
plt.title("Test Görseli")
plt.axis('off')
plt.show()
