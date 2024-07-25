import tensorflow as tf
import cv2
import numpy as np


def load_model():
    model = tf.keras.models.load_model("custom_model.h5")
    return model


if __name__ == "__main__":
    model = tf.keras.models.load_model("custom_model.h5")
    image = cv2.imread("archive (16)/Braille Dataset/Braille Dataset/g1.JPG11dim.jpg")
    image = np.array(image)
    image = image / 255.0
    # tf.expand_dims(X_test[0], axis=0)
    temp = model.predict(tf.expand_dims(image, axis=0))
    ans = np.argmax(temp)
    print(ans)
