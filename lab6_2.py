import numpy as np
from tensorflow import keras
from PIL import Image

CLASSES = ["circle", "square", "triangle"]
IMG_SIZE = 64

model = keras.models.load_model("shapes_cnn.h5")
print("Модель загружена.")

def load_image(path):
    img = Image.open(path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype("float32") / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return img

try:
    path = input("Введите путь к изображению (PNG/JPG): ")
    img = load_image(path)

    prediction = model.predict(img)
    cls = int(np.argmax(prediction))

    print(f"Предсказанный класс: {cls} ({CLASSES[cls]})")

except Exception as e:
    print("Ошибка:", e)
