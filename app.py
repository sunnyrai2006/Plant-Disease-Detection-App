import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import gdown

model_file = "plant_model.h5"
file_id = "1v7uboHqlTbEkZa9IPTTj34ylkl5r-4DN"

if not os.path.exists(model_file):
    st.info("Downloading model from Google Drive")
    url = "https://drive.google.com/file/d/1v7uboHqlTbEkZa9IPTTj34ylkl5r-4DN/view?usp=sharing"
    gdown.download(url,model_file,quiet=False)
    st.success("model downloaded")

model = tf.keras.models.load_model(model_file, compile=False)
with open("class_labels.json") as f:
    class_labels = json.load(f)

labels = {v:k for k,v in class_labels.items()}
st.set_page_config(
    page_title= "Plant Disease Detection App", 
    page_icon="🍀",
    layout="wide")
st.title("🌱 Plant Disease Detection")
st.write("Upload a leaf image")

uploaded_file = st.file_uploader(
    "Choose image",
    type=["jpg","jpeg","png"]
)
def predict(img):
    img = img.resize((224,224))
    img_arr = np.array(img)/255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    preds = model.predict(img_arr)
    idx = np.argmax(preds)
    return labels[idx], 

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
    if st.button("Predict"):
        result = predict(image)
        st.success(f"Disease: {result}")