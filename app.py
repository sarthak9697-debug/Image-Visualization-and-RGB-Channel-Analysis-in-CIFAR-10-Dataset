from pathlib import Path
import io

import numpy as np
import streamlit as st
from PIL import Image

APP_BUILD = "CIFAR10-CAMERA-SIDEBYSIDE-2026-11-06"

CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)
SAMPLE_CLASSES = (
    "automobile", "bird", "cat", "deer",
    "frog", "horse", "ship", "truck"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.keras"
SAMPLE_DIR = BASE_DIR / "sample_images"

st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🖼️", layout="wide")


@st.cache_data
def load_local_samples():
    samples = {}
    for class_name in SAMPLE_CLASSES:
        image_path = SAMPLE_DIR / f"{class_name}.png"
        if not image_path.is_file():
            available = sorted(p.name for p in SAMPLE_DIR.glob("*") if p.is_file()) if SAMPLE_DIR.exists() else []
            raise FileNotFoundError(
                f"Missing sample image: sample_images/{class_name}.png. "
                f"Expected folder: {SAMPLE_DIR}. Files found: {available or 'none'}. "
                "Commit the complete sample_images folder beside app.py."
            )
        with Image.open(image_path) as source:
            samples[class_name] = source.convert("RGB").copy()
    return samples


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("model.keras is missing. Place it beside app.py.")
    import tensorflow as tf
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


def prepare_for_model(pil_image):
    image = pil_image.convert("RGB").resize((32, 32), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict(model, pil_image):
    output = np.asarray(model.predict(prepare_for_model(pil_image), verbose=0))[0]
    if output.shape[0] != len(CLASS_NAMES):
        raise ValueError(f"Model returned {output.shape[0]} outputs; expected 10.")
    return int(np.argmax(output)), output


def display_result(image, scores, predicted_id, ground_truth=None):
    predicted_name = CLASS_NAMES[predicted_id]
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Input image")
        st.image(image, use_container_width=True)
        if ground_truth:
            st.caption(f"Ground-truth label: **{ground_truth.title()}**")
    with right:
        st.subheader("Prediction")
        st.success(f"**{predicted_name.title()}**")
        st.metric("Confidence", f"{float(scores[predicted_id]) * 100:.2f}%")
        if ground_truth:
            if predicted_name == ground_truth:
                st.success("Prediction matches the sample label.")
            else:
                st.warning("Prediction differs from the sample's ground-truth label.")
        st.markdown("**Class probabilities**")
        st.bar_chart({CLASS_NAMES[i]: float(scores[i]) for i in range(len(CLASS_NAMES))})


st.title("CIFAR-10 Image Classification")
st.caption("CNN image classifier · 10 prediction categories")
st.caption(f"App build: `{APP_BUILD}`")
st.info(
    "Eight sample images are bundled locally. No CIFAR-10 dataset download is needed. "
    "You can select a sample, upload an image, or take a photo with your camera."
)

try:
    samples = load_local_samples()
except Exception as error:
    st.error(f"Sample gallery setup error: {error}")
    st.stop()

st.header("1. Local sample gallery")
gallery_cols = st.columns(4)
for index, class_name in enumerate(SAMPLE_CLASSES):
    with gallery_cols[index % 4]:
        st.image(samples[class_name], caption=class_name.title(), use_container_width=True)

st.divider()
st.header("2. Test the model")
mode = st.radio(
    "Choose how to provide an image",
    ["Local sample", "Upload image", "Take a snapshot"],
    horizontal=True,
    key="input_mode_camera",
)

chosen_image = None
ground_truth = None
auto_predict = False

if mode == "Local sample":
    selected_class = st.selectbox(
        "Choose a sample category",
        options=SAMPLE_CLASSES,
        format_func=lambda value: value.title(),
        key="sample_selector_camera",
    )
    chosen_image = samples[selected_class].copy()
    ground_truth = selected_class

elif mode == "Upload image":
    upload = st.file_uploader(
        "Upload JPG, JPEG or PNG",
        type=["jpg", "jpeg", "png"],
        key="upload_camera",
    )
    if upload is not None:
        try:
            chosen_image = Image.open(io.BytesIO(upload.getvalue())).convert("RGB")
        except Exception as error:
            st.error(f"Could not open image: {error}")

else:
    camera_photo = st.camera_input("Take a snapshot")
    if camera_photo is not None:
        try:
            chosen_image = Image.open(io.BytesIO(camera_photo.getvalue())).convert("RGB")
            auto_predict = True
        except Exception as error:
            st.error(f"Could not read camera snapshot: {error}")

if chosen_image is not None:
    # A camera capture is predicted automatically after the camera's Take Photo click.
    if auto_predict:
        should_predict = True
    else:
        should_predict = st.button("Run prediction", type="primary", key="predict_camera")

    if should_predict:
        try:
            with st.spinner("Loading the model and running prediction..."):
                model = load_model()
                predicted_id, scores = predict(model, chosen_image)
            st.session_state["last_prediction_camera"] = {
                "image": chosen_image.copy(),
                "predicted_id": predicted_id,
                "scores": scores.tolist(),
                "ground_truth": ground_truth,
            }
        except Exception as error:
            st.error(f"Prediction failed: {error}")

# Keep the latest result visible on the page, with input and prediction side by side.
result = st.session_state.get("last_prediction_camera")
if result:
    st.divider()
    st.header("3. Prediction result")
    display_result(
        result["image"],
        np.asarray(result["scores"], dtype=float),
        int(result["predicted_id"]),
        result["ground_truth"],
    )

st.divider()
st.subheader("Input specifications")
st.markdown(
    "- Accepted formats: JPG, JPEG, PNG, and camera snapshots\n"
    "- Colour channels: RGB (grayscale converted to RGB)\n"
    "- Model input: 32 × 32 × 3\n"
    "- Pixel values: normalized to 0–1\n"
    "- Prediction classes: all 10 CIFAR-10 categories\n"
    "- Built-in gallery: 8 locally stored categories"
)
st.caption(
    "CIFAR-10 images are low resolution. Predictions on unrelated high-resolution "
    "photographs may be unreliable. This demo is for educational use."
)
