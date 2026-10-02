import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# 1. Load your model (adjust file name/path if needed)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

st.title("Image Classification Model")

# 2. File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # ==========================================
    # ADD THE FIX RIGHT HERE (PREPROCESSING STEP)
    # ==========================================
    image = Image.open(uploaded_file)
    
    # Converts 4-channel (RGBA) or 1-channel (grayscale) to 3-channel RGB
    image = image.convert("RGB")
    
    # Resize to match your model's expected input dimensions (224, 224)
    image = image.resize((224, 224))
    
    # Convert PIL Image to NumPy array
    img_array = np.array(image, dtype=np.float32)
    
    # Normalize if your training pipeline used normalization (e.g., 0-1 range)
    # img_array = img_array / 255.0
    
    # Add batch dimension: changes shape from (224, 224, 3) -> (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # ==========================================

    # 3. Model Prediction
    if st.button("Predict"):
        prediction = model.predict(img_array)
        st.write("Prediction output:", prediction)