import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tensorflow import keras

from PIL import Image


def predict_character(image):
    image = cv2.resize(image, (28, 28))
    print("image", image.shape)
    image = np.array(image)
    print("image", image.shape)
    image = image / 255.0
    # tf.expand_dims(X_test[0], axis=0)
    temp = model.predict(tf.expand_dims(image, axis=0))
    ans = np.argmax(temp)
    ans = int_to_telugu[ans]
    return ans


"""
"""


def main():
    st.markdown(
        "<h1 style='text-align: center;'>Braille Image to Telugu</h1>",
        unsafe_allow_html=True,
    )
    page = st.sidebar.selectbox("Select a page", ["Home", "Detect"])

    if page == "Home":
        st.markdown(
            "<h2 style='color: red;'>Introduction:</h2>", unsafe_allow_html=True
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                """
            <style>
                .content {
                    font-size: 20px;
                    line-height: 1.6;
                    font-family: Arial, sans-serif;
                    background-color: #f9f9f9;
                    padding: 20px;
                    border-radius: 8px;
                    max-width: 800px;
                    margin: auto;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                }

                .content h1 {
                    font-size: 26px;
                    text-align: left;
                    color: #333;
                }

                .content li {
                    margin: 15px 0;
                    color: #555;
                    list-style-type: disc;
                    margin-left: 20px;
                }
            </style>

            <div class="content">
                <h1>Understanding Braille</h1>
                <ul>
                    <li>Braille is a tactile writing system used by people who are blind or visually impaired.</li>
                    <li>It was invented by Louis Braille in the early 19th century and has since been adapted for numerous languages worldwide.</li>
                    <li>Braille literacy is crucial for the education, communication, and daily activities of blind individuals, providing them with access to information and opportunities.</li>
                    <li>Each Braille character is made up of a combination of up to six dots in a 3x2 grid.</li>
                    <li>Braille script consists of patterns of raised dots arranged in cells or grids that represent letters, numbers, and punctuation.</li>
                </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            image = Image.open("home_image.jpg")
            st.image(image, caption="Braille Literacy", use_column_width=True)
        st.markdown(
            "<h2 style='color: red;'>Bharati Braille and Telugu Braille:</h2>",
            unsafe_allow_html=True,
        )
        st.write(
            """
            Bharati Braille, or Bharatiya Braille, is a largely unified braille script for writing the languages of India.
            Telugu Braille is one of the Bharati braille alphabets, and it largely conforms to the letter values of the other Bharati alphabets.
            """
        )
        st.image(
            "info.jpg",
            caption="Braille to Telugu Characters",
        )

    elif page == "Detect":
        st.write("Braille image to telugu")
        uploaded_image = st.file_uploader("Upload a image", type=[".jpg"])
        if uploaded_image is not None:
            image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
            ans = predict_character(image)
            st.header("The character is:   " + ans)


if __name__ == "__main__":
    model = tf.keras.models.load_model("custom_model.h5")
    int_to_telugu = {
        0: "అ",
        1: "ఆ",
        2: "ఇ",
        3: "ఈ",
        4: "ఉ",
        5: "ఊ",
        6: "ఋ",
        7: "ౠ",
        8: "ఎ",
        9: "ఏ",
        10: "ఐ",
        11: "ఒ",
        12: "ఓ",
        13: "ఔ",
        14: "అం",
        15: "అః",
        16: "క",
        17: "ఖ",
        18: "గ",
        19: "ఘ",
        20: "ఙ",
        21: "చ",
        22: "ఛ",
        23: "జ",
        24: "ఝ",
        25: "ఞ",
        26: "ట",
        27: "ఠ",
        28: "డ",
        29: "ఢ",
        30: "ణ",
        31: "త",
        32: "థ",
        33: "ద",
        34: "ధ",
        35: "న",
        36: "ప",
        37: "ఫ",
        38: "బ",
        39: "భ",
        40: "మ",
        41: "య",
        42: "ర",
        43: "ల",
        44: "వ",
        45: "శ",
        46: "ష",
        47: "స",
        48: "హ",
        49: "ళ",
        50: "క్ష",
        51: "ఱ",
    }
    main()
