import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# 1. Load the model once into memory
@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model("model.keras")

model = load_keras_model()

# 2. UI Layout
st.set_page_config(page_title="Image Classifier", layout="centered")
st.title("Keras Image Classifier")
st.write("Upload an image to get a classification prediction.")

# 3. File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify Image"):
        # Preprocess to match training dimensions (e.g., 224x224 RGB)
        target_size = (224, 224)
        image = image.resize(target_size)
        img_array = np.array(image, dtype=np.float32)

        # Handle grayscale vs RGB if necessary
        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)

        # Normalize pixel values if your model expects [0, 1]
        img_array = img_array / 255.0

        # Expand dimensions to create batch: (1, 224, 224, 3)
        input_tensor = np.expand_dims(img_array, axis=0)

        # Inference
        predictions = model.predict(input_tensor)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))

        st.success(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2%})")