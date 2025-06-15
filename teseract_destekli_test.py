import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle

# Tesseract yolu (eğer sistemde ayarlanmadıysa)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Dosya Yolları ---
model_path = r"C:\Users\rezmi\Desktop\Dil_modeli\osmanli_model_128.h5"
encoder_path = r"C:\Users\rezmi\Desktop\Dil_modeli\label_encoder_128.pkl"
test_image_path = r"C:\Users\rezmi\Desktop\Dil_modeli\test.png"

# --- Model ve encoder yükle ---
model = load_model(model_path)
with open(encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# --- Görseli yükle ve griye çevir ---
img = cv2.imread(test_image_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_copy = img.copy()

# --- OCR ile karakter kutuları al ---
custom_config = r'--oem 3 --psm 6'  # LSTM + kelime satırı
ocr_data = pytesseract.image_to_data(img_gray, lang='ara', config=custom_config, output_type=pytesseract.Output.DICT)

tahminler = []
n_boxes = len(ocr_data['text'])

for i in range(n_boxes):
    if int(ocr_data['conf'][i]) > 50:  # Güven skoru yüksek olanları al
        (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
        roi = img_gray[y:y+h, x:x+w]

        if roi.shape[0] < 10 or roi.shape[1] < 10:
            continue  # Gürültü

        roi_resized = cv2.resize(roi, (128, 128)) / 255.0
        roi_reshaped = roi_resized.reshape(1, 128, 128, 1)

        prediction = model.predict(roi_reshaped, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        tahminler.append(predicted_label)

        # Kutuyu çiz
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_copy, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)

# --- Sonuçları göster ---
print("🟩 Algılanan harf sayısı:", len(tahminler))
print("🔡 Tahmin edilen harfler:", "".join(tahminler))

# --- Görselleştirme ---
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
plt.title("Tahmin Edilen Harfler ve Konumları")
plt.axis('off')
plt.show()
