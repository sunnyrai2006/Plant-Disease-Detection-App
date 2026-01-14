import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

model_file = "plant_model.h5"

if not os.path.exists(model_file):
    st.error("Model file not found! Make sure 'plant_model.h5' is in the repo.")
else:
    model = tf.keras.models.load_model(model_file, compile=False)
    with open("class_labels.json") as f:
        class_labels = json.load(f)

    labels = {v: k for k, v in class_labels.items()}
st.set_page_config(
    page_title="Plant Disease Detection App",
    page_icon="🍀",
    layout="wide"
)

st.title("🌱 Plant Disease Detection")
st.write("Upload a leaf image to detect its disease:")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

def predict(img):
    img = img.resize((224, 224)) 
    img_arr = np.array(img) / 255.0  
    img_arr = np.expand_dims(img_arr, axis=0) 
    preds = model.predict(img_arr)
    idx = np.argmax(preds)
    confidence = round(100 * np.max(preds), 2)
    return labels[idx], confidence


if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Predict"):
        result, confidence = predict(image)
        st.success(f"Disease: {result}")
        st.info(f"Confidence: {confidence}%")