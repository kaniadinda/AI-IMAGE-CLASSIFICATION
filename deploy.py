import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import gdown
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import load_img, img_to_array

img_size = 48

MODEL_PATH = "AIGeneratedModel.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1uQQZvi9p5ZTKCr7ZDImBV7z3ox1D_AZd"
    gdown.download(url, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)

st.title("AI Image Classifier")       
        
img = st.file_uploader("Upload your Image")

if img is not None and st.button("Check"):
    image = Image.open(img).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    image = ImageOps.fit(image, (48, 48), Image.Resampling.LANCZOS)
    img_array = np.array(image)
    img_array = img_array / 255.0
    test = np.expand_dims(img_array, axis=0)
    y = model.predict(test)

    if y[0][0] <= 0.5:
        st.success("The image is **Real**.")
    else:
        st.error("The image is **AI Generated**.")
