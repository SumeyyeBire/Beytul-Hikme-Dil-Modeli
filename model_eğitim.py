import pandas as pd
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pickle

# --- VERİ YÜKLEME (DataFrame olarak) ---
csv_path = r"C:\Users\rezmi\Desktop\Dil_modeli\artirilmis_veri\labels.csv"
img_dir = r"C:\Users\rezmi\Desktop\Dil_modeli\artirilmis_veri\images"

df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")

# --- Etiketleri Önce Encoder ile Sayısal Hale Getir ---
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['text'])
num_classes = len(label_encoder.classes_)

# --- Eğitim ve Test Bölme ---
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_encoded'])

# --- Data Generator Tanımı ---
class DataGenerator(Sequence):
    def __init__(self, df, img_dir, batch_size=8, dim=(1024,1024), shuffle=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = np.empty((self.batch_size, *self.dim, 1), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)

        for i, idx in enumerate(batch_indexes):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row['file'])
            img = Image.open(img_path).convert('L').resize(self.dim)
            X[i, :, :, 0] = np.array(img, dtype=np.float32) / 255.0
            y[i] = row['label_encoded']

        return X, to_categorical(y, num_classes=num_classes)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# --- Generator Oluştur ---
train_gen = DataGenerator(df_train, img_dir, batch_size=8, dim=(1024, 1024), shuffle=True)
val_gen = DataGenerator(df_val, img_dir, batch_size=8, dim=(1024, 1024), shuffle=False)

print("Veri hazır, sınıf sayısı:", num_classes)

# --- MODEL TANIMI ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(1024, 1024, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- MODELİ EĞİTME ---
model.fit(train_gen, validation_data=val_gen, epochs=30)

# --- MODELİ VE ETİKET DÖNÜŞTÜRÜCÜSÜNÜ KAYDET ---
model_path = r"C:\Users\rezmi\Desktop\Dil_modeli\osmanli_model_1024.h5"
pkl_path = r"C:\Users\rezmi\Desktop\Dil_modeli\label_encoder_1024.pkl"

model.save(model_path)

with open(pkl_path, "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Eğitim tamamlandı.")
print("💾 Model ve label_encoder dosyası başarıyla kaydedildi.")
