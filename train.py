import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# =========================
# Dataset Path
# =========================

dataset_path = "dataset"


if not os.path.exists(dataset_path):
    print("Dataset folder not found")
    exit()


print("Dataset folders:")
print(os.listdir(dataset_path))


# =========================
# Image Preprocessing
# =========================

data = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.5
)


# =========================
# Training Data
# =========================

train_data = data.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=2,
    class_mode="binary",
    subset="training",
    shuffle=True
)


# =========================
# Validation Data
# =========================

validation_data = data.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=2,
    class_mode="binary",
    subset="validation"
)


print("Classes:")
print(train_data.class_indices)

print("Training images:", train_data.samples)
print("Validation images:", validation_data.samples)



# =========================
# Save Dataset Image Preview
# =========================

images, labels = next(train_data)


plt.figure(figsize=(8,8))


for i in range(len(images)):

    plt.subplot(2,2,i+1)

    plt.imshow(images[i])


    if labels[i] == 0:
        plt.title("Diseased")
    else:
        plt.title("Healthy")


    plt.axis("off")


plt.savefig("dataset_preview.png")

plt.close()


print("Dataset image saved as dataset_preview.png")



# =========================
# 7 Layer CNN Model
# =========================

model = Sequential()


# Layer 1
model.add(
    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(128,128,3)
    )
)


# Layer 2
model.add(
    MaxPooling2D(2,2)
)


# Layer 3
model.add(
    Conv2D(
        64,
        (3,3),
        activation="relu"
    )
)


# Layer 4
model.add(
    MaxPooling2D(2,2)
)


# Layer 5
model.add(
    Flatten()
)


# Layer 6
model.add(
    Dense(
        128,
        activation="relu"
    )
)


model.add(
    Dropout(0.5)
)


# Layer 7
model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)



# =========================
# Compile
# =========================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()



# =========================
# Training
# =========================

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=20
)



# =========================
# Save Model
# =========================

model.save(
    "plant_disease_cnn.h5"
)


print("Training Completed Successfully")