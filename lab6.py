import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageDraw

# Классы
CLASSES = ["circle", "square", "triangle"]
IMG_SIZE = 64

# генерация изображений
def draw_shape(shape):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
    d = ImageDraw.Draw(img)
    m = 10

    if shape == "circle":
        d.ellipse([m, m, IMG_SIZE - m, IMG_SIZE - m], outline=0, width=6)
    elif shape == "square":
        d.rectangle([m, m, IMG_SIZE - m, IMG_SIZE - m], outline=0, width=6)
    elif shape == "triangle":
        d.polygon([(IMG_SIZE//2, m), (m, IMG_SIZE - m), (IMG_SIZE - m, IMG_SIZE - m)], outline=0, width=6)

    return np.array(img, dtype=np.uint8)

x_train, y_train = [], []
x_test, y_test = [], []

for label, name in enumerate(CLASSES):
    for _ in range(2000):
        x_train.append(draw_shape(name))
        y_train.append(label)
    for _ in range(400):
        x_test.append(draw_shape(name))
        y_test.append(label)

x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0 # нормализуем пиксели (к 0,1)
x_test  = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
y_train = np.array(y_train)
y_test  = np.array(y_test)


model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"), # окно 3на3, 32 фильтра (карты признаков), делаем модель налинейной
    layers.MaxPooling2D((2, 2)), #Уменьшает размер карты признаков в 2 раза по ширине и высоте.

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)), #Второй сверточный слой “смотрит” уже не на пиксели, а на признаки из первого слоя.

    layers.Flatten(), #3 канала в один вектор
    layers.Dense(64, activation="relu"), # сам классификатор. берёт все найденные признаки и учится их комбинировать в решение.
    layers.Dense(3, activation="softmax") #Финальный слой: 3 нейрона = 3 класса (circle/square/triangle).


])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1)

loss, acc = model.evaluate(x_test, y_test)
print(f"Точность на тестовой выборке: {acc:.4f}")

model.save("shapes_cnn2.h5")
print("Модель сохранена")

# генерируем файл для проверки
Image.fromarray(draw_shape("circle")).save("sample_circle.png")
print("Тестовый файл: sample_circle.png")
