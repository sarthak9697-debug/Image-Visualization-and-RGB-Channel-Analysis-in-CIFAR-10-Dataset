from pathlib import Path
import io

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

APP_BUILD = "CIFAR10-LABEL-FIX-2026-11-06"
CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)
MODEL_PATH = Path(__file__).resolve().parent / "model.keras"

st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🖼️", layout="wide")

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing {MODEL_PATH.name}. Put it beside app.py.")
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)

@st.cache_data(show_spinner=False)
def download_test_set():
    # x_test[i] and y_test[i] are a paired image/label from the official Keras CIFAR-10 test set.
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return x_test, y_test.reshape(-1).astype(np.int64)

def one_sample_for_each_label(x, y):
    samples = {}
    for label_id in range(len(CLASS_NAMES)):
        positions = np.where(y == label_id)[0]
        if positions.size == 0:
            raise RuntimeError(f"No CIFAR-10 test image found for label ID {label_id}.")
        idx = int(positions[0])
        # Keep the image and its corresponding label together in the same record.
        samples[label_id] = {
            "image": x[idx].copy(),
            "label_id": int(y[idx]),
            "dataset_index": idx,
        }
        if samples[label_id]["label_id"] != label_id:
            raise RuntimeError("Image/label pairing validation failed.")
    return samples

def prepare_for_model(pil_image):
    pil_image = pil_image.convert("RGB").resize((32, 32), Image.Resampling.LANCZOS)
    array = np.asarray(pil_image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)

def predict(model, pil_image):
    output = np.asarray(model.predict(prepare_for_model(pil_image), verbose=0))[0]
    if output.shape[0] != len(CLASS_NAMES):
        raise ValueError(
            f"Model returned {output.shape[0]} outputs; expected 10 CIFAR-10 classes."
        )
    predicted_id = int(np.argmax(output))
    return predicted_id, output

st.title("CIFAR-10 Image Classification")
st.caption("CNN image classifier · 10 CIFAR-10 categories")
st.caption(f"App build: `{APP_BUILD}`")
st.info(
    "The sample gallery displays official CIFAR-10 test images. Each sample's label is read "
    "from the label array at the exact same dataset index as its image."
)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load model.keras: {error}")
    st.stop()

st.header("1. Correctly labelled sample gallery")
if st.button("Download / refresh the 10 sample images", type="primary"):
    try:
        x_test, y_test = download_test_set()
        st.session_state["cifar10_samples_v2"] = one_sample_for_each_label(x_test, y_test)
        st.session_state.pop("chosen_sample_v2", None)
    except Exception as error:
        st.error(f"Could not load CIFAR-10 test data: {error}")

samples = st.session_state.get("cifar10_samples_v2")

if samples:
    gallery_cols = st.columns(5)
    for label_id in range(10):
        record = samples[label_id]
        # Use the stored label paired with this image, not a separately assigned caption.
        actual_label_id = record["label_id"]
        actual_label = CLASS_NAMES[actual_label_id]
        with gallery_cols[label_id % 5]:
            st.image(
                record["image"],
                caption=f"{actual_label.title()} · label ID {actual_label_id}",
                use_container_width=True
            )
            st.caption(f"Test-set item #{record['dataset_index']}")
else:
    st.warning("Click the button above to load the official CIFAR-10 test samples.")

st.divider()
st.header("2. Test the model")
mode = st.radio(
    "Input type",
    ["Use one of the labelled CIFAR-10 samples", "Upload an image"],
    key="input_mode_v2"
)

chosen_image = None
ground_truth = None

if mode == "Use one of the labelled CIFAR-10 samples":
    if samples:
        sample_label_id = st.selectbox(
            "Choose a sample category",
            options=list(range(10)),
            format_func=lambda i: (
                f"Sample label {i}: {CLASS_NAMES[samples[i]['label_id']].title()}"
            ),
            key="sample_selector_v2"
        )
        selected_record = samples[sample_label_id]
        chosen_image = Image.fromarray(selected_record["image"])
        ground_truth = CLASS_NAMES[selected_record["label_id"]]
        st.image(
            chosen_image,
            caption=f"Selected sample — dataset ground truth: {ground_truth.title()}",
            width=260
        )
    else:
        st.info("Load the sample gallery first.")
else:
    upload = st.file_uploader(
        "Upload JPG, JPEG or PNG",
        type=["jpg", "jpeg", "png"],
        key="upload_v2"
    )
    if upload is not None:
        try:
            chosen_image = Image.open(io.BytesIO(upload.getvalue())).convert("RGB")
            st.image(chosen_image, caption="Uploaded image", width=260)
        except Exception as error:
            st.error(f"Could not open image: {error}")

if chosen_image is not None and st.button("Run prediction", type="primary", key="predict_v2"):
    try:
        predicted_id, scores = predict(model, chosen_image)
        predicted_name = CLASS_NAMES[predicted_id]
        st.subheader("Prediction")
        st.success(f"Predicted class: **{predicted_name.title()}**")
        st.metric("Model confidence", f"{float(scores[predicted_id]) * 100:.2f}%")
        if ground_truth is not None:
            st.write(f"**Actual dataset label:** {ground_truth.title()}")
            if predicted_name == ground_truth:
                st.success("Prediction matches the sample's ground-truth label.")
            else:
                st.warning("Prediction differs from the ground-truth label; this is a model error, not a sample-label mismatch.")
        st.bar_chart({CLASS_NAMES[i]: float(scores[i]) for i in range(10)})
    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
st.subheader("Input specifications")
st.markdown(
    "- Accepted formats: JPG, JPEG, PNG\n"
    "- Colour channels: RGB (grayscale converted to RGB)\n"
    "- Model input: 32 × 32 × 3\n"
    "- Pixel values: normalized to 0–1\n"
    "- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck"
)
st.caption(
    "CIFAR-10 is a low-resolution image dataset. Predictions on unrelated high-resolution "
    "photographs may be unreliable. This demo is for education, not safety-critical use."
)
