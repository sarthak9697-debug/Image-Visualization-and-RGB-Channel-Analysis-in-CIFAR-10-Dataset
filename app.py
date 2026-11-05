import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🖼️", layout="wide")

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
MODEL_PATH = Path(__file__).parent / "model.keras"

@st.cache_resource(show_spinner="Loading the trained CNN model...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.keras was not found beside app.py.")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((32, 32), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@st.cache_data(show_spinner=False)
def get_sample_data():
    # Intentionally called only after the user requests sample images.
    # This avoids blocking the initial Streamlit page load.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return x_test[:100], y_test[:100].reshape(-1)

def predict(model, batch):
    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    return idx, probs

st.title("🖼️ CIFAR-10 Image Classification")
st.caption("A convolutional neural network that classifies 32 × 32 colour images into 10 CIFAR-10 categories.")
with st.expander("About this project", expanded=True):
    st.write(
        "Upload an image or explicitly load a small set of CIFAR-10 test examples. "
        "The model expects RGB images and automatically resizes them to 32 × 32 pixels."
    )
    st.info("This is an educational image-classification demo, not a general-purpose vision system. Best results are expected for CIFAR-10-like images.")

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("1. Choose an input image")
    mode = st.radio("Input source", ["Upload an image", "Use CIFAR-10 sample"], horizontal=False)
    image = None
    true_label = None
    if mode == "Upload an image":
        uploaded = st.file_uploader("Upload JPG, JPEG, or PNG", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            try:
                image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
            except Exception as exc:
                st.error(f"Unable to read image: {exc}")
    else:
        st.caption("Sample images are fetched only when you click the button. The first download may take a little time.")
        if st.button("Load CIFAR-10 sample images", type="primary"):
            try:
                st.session_state["cifar_samples"] = get_sample_data()
            except Exception as exc:
                st.error(f"Could not load CIFAR-10 samples: {exc}")
        if "cifar_samples" in st.session_state:
            images, labels = st.session_state["cifar_samples"]
            choices = list(range(len(images)))
            selected = st.selectbox(
                "Select a sample",
                choices,
                format_func=lambda i: f"Sample {i + 1} — actual label: {CLASS_NAMES[int(labels[i])]}"
            )
            image = Image.fromarray(images[selected])
            true_label = CLASS_NAMES[int(labels[selected])]
    if image is not None:
        st.image(image, caption="Selected input image", use_container_width=True)

with right:
    st.subheader("2. Model prediction")
    if image is not None:
        batch = prepare_image(image)
        with st.spinner("Classifying image..."):
            predicted_idx, probabilities = predict(model, batch)
        st.success(f"Predicted class: **{CLASS_NAMES[predicted_idx].title()}**")
        if true_label is not None:
            st.write(f"Actual CIFAR-10 label: **{true_label.title()}**")
        st.metric("Model confidence", f"{float(probabilities[predicted_idx]) * 100:.1f}%")
        st.write("Class probabilities")
        st.bar_chart({CLASS_NAMES[i]: float(probabilities[i]) for i in range(10)})
    else:
        st.write("Choose an image on the left to see its prediction.")

st.divider()
st.subheader("Model input requirements")
st.write("- Image type: JPG, JPEG, or PNG")
st.write("- Colour mode: RGB (grayscale images are converted to RGB)")
st.write("- Model input: 32 × 32 × 3 pixels, values normalized to 0–1")
st.write("- Supported classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck")
st.caption("Note: CIFAR-10 sample loading uses the Keras dataset downloader when explicitly requested. Uploading an image does not require downloading CIFAR-10.")
