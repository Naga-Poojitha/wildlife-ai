import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("wildlife_model.keras")

# Load classes
with open("classes.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

st.set_page_config(page_title="Wildlife AI", layout="centered")

st.title("🐾 Wildlife Animal Classification")
st.write("Upload an image to identify the animal")

uploaded_file = st.file_uploader("Choose an animal image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224,224))
    img_array = np.expand_dims(np.array(img), axis=0)

    pred = model.predict(img_array)
    confidence = np.max(pred)*100
    predicted_class = class_names[np.argmax(pred)]

    st.subheader("Prediction")
    st.success(f"Animal: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("Conservation Message")
    st.write(f"{predicted_class} species are important for ecosystem balance. Protecting them helps preserve biodiversity.")
