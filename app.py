from pathlib import Path
import io

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

APP_BUILD = "CIFAR10-LOCAL-SAMPLES-8-2026-11-06"

# The classifier retains the full 10-class CIFAR-10 output mapping.
CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)
# Local gallery contains the eight classes available in the supplied notebook image.
SAMPLE_CLASSES = (
    "automobile", "bird", "cat", "deer",
    "frog", "horse", "ship", "truck"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.keras"
SAMPLE_DIR = BASE_DIR / "sample_images"

st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🖼️", layout="wide")


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("model.keras is missing. Place it beside app.py.")
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


@st.cache_data
def load_local_samples():
    samples = {}
    for class_name in SAMPLE_CLASSES:
        image_path = SAMPLE_DIR / f"{class_name}.png"
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing sample image: sample_images/{class_name}.png"
            )
        with Image.open(image_path) as source:
            samples[class_name] = source.convert("RGB").copy()
    return samples


def prepare_for_model(pil_image):
    image = pil_image.convert("RGB").resize((32, 32), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict(model, pil_image):
    output = np.asarray(
        model.predict(prepare_for_model(pil_image), verbose=0)
    )[0]
    if output.shape[0] != len(CLASS_NAMES):
        raise ValueError(
            f"Model returned {output.shape[0]} outputs; expected 10 CIFAR-10 classes."
        )
    predicted_id = int(np.argmax(output))
    return predicted_id, output


st.title("CIFAR-10 Image Classification")
st.caption("CNN image classifier · 10 prediction categories")
st.caption(f"App build: `{APP_BUILD}`")
st.info(
    "Eight correctly labelled sample images are stored in the local "
    "`sample_images/` folder. The gallery loads immediately without downloading "
    "the CIFAR-10 dataset. The trained model still predicts across all 10 classes."
)

try:
    model = load_model()
    samples = load_local_samples()
except Exception as error:
    st.error(f"Application setup error: {error}")
    st.stop()

st.header("1. Local sample gallery")
gallery_cols = st.columns(4)
for index, class_name in enumerate(SAMPLE_CLASSES):
    with gallery_cols[index % 4]:
        st.image(
            samples[class_name],
            caption=f"{class_name.title()} · ground-truth label",
            use_container_width=True,
        )

st.divider()
st.header("2. Test the model")
mode = st.radio(
    "Input type",
    ["Use a local sample image", "Upload an image"],
    key="input_mode_local",
)

chosen_image = None
ground_truth = None

if mode == "Use a local sample image":
    selected_class = st.selectbox(
        "Choose a sample category",
        options=SAMPLE_CLASSES,
        format_func=lambda value: value.title(),
        key="sample_selector_local",
    )
    chosen_image = samples[selected_class].copy()
    ground_truth = selected_class
    st.image(
        chosen_image,
        caption=f"Selected sample · ground-truth label: {ground_truth.title()}",
        width=260,
    )
else:
    upload = st.file_uploader(
        "Upload JPG, JPEG or PNG",
        type=["jpg", "jpeg", "png"],
        key="upload_local",
    )
    if upload is not None:
        try:
            chosen_image = Image.open(io.BytesIO(upload.getvalue())).convert("RGB")
            st.image(chosen_image, caption="Uploaded image", width=260)
        except Exception as error:
            st.error(f"Could not open image: {error}")

if chosen_image is not None and st.button(
    "Run prediction", type="primary", key="predict_local"
):
    try:
        predicted_id, scores = predict(model, chosen_image)
        predicted_name = CLASS_NAMES[predicted_id]
        st.subheader("Prediction")
        st.success(f"Predicted class: **{predicted_name.title()}**")
        st.metric("Model confidence", f"{float(scores[predicted_id]) * 100:.2f}%")
        if ground_truth is not None:
            st.write(f"**Sample ground-truth label:** {ground_truth.title()}")
            if predicted_name == ground_truth:
                st.success("Prediction matches the sample label.")
            else:
                st.warning(
                    "Prediction differs from the sample label. This is a model "
                    "classification error, not a gallery-label mismatch."
                )
        st.bar_chart(
            {CLASS_NAMES[i]: float(scores[i]) for i in range(len(CLASS_NAMES))}
        )
    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
st.subheader("Input specifications")
st.markdown(
    "- Accepted formats: JPG, JPEG, PNG\n"
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
