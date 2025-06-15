import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

# CSV dosyasını oku (önemli: ayırıcı ve encoding)
def new_func():
    df = pd.read_csv(
    r"C:\Users\rezmi\Desktop\Dil_modeli\artirilmis_veri\labels.csv",
    sep=";",  # <- Noktalı virgülü ayırıcı olarak tanımla
    encoding="utf-8-sig"
)
    return df

try:
    df = new_func()


except Exception as e:
    print("CSV okunurken hata oluştu:", e)
    exit()

# Sütun adlarını kontrol et
print("Sütunlar:", df.columns)

# image_folder: Görsellerin bulunduğu klasör
image_folder = r"C:\Users\rezmi\Desktop\Dil_modeli\artirilmis_veri\images"  # örneğin: ./images/1.png, ./images/2.png

# İlk 5 görseli ve etiketlerini göster
for i in range(min(15000, len(df))):
    row = df.iloc[i]

    try:
        img_file = row['file']
        label = row['text']
    except KeyError:
        print("CSV'de 'file' veya 'text' sütunu eksik.")
        break

    img_path = os.path.join(image_folder, img_file)

    if os.path.exists(img_path):
        try:
            img = Image.open(img_path)
            plt.imshow(img)
            plt.title(f"Etiket: {label}")
            plt.axis("off")
            plt.show()
        except Exception as e:
            print(f"Görsel açılırken hata oluştu ({img_file}):", e)
    else:
        print(f"Görsel bulunamadı: {img_path}")
