import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


# Load trained model

model = tf.keras.models.load_model(
    "plant_disease_cnn.h5"
)


# Image path to test

img_path = "test.jpg"


# Load image

img = image.load_img(
    img_path,
    target_size=(128,128)
)


# Convert image to array

img_array = image.img_to_array(img)


# Add batch dimension

img_array = np.expand_dims(
    img_array,
    axis=0
)


# Normalize

img_array = img_array / 255.0



# Prediction

prediction = model.predict(img_array)


if prediction[0][0] > 0.5:
    print("Prediction: Healthy Plant")
else:
    print("Prediction: Diseased Plant")