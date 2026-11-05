from pathlib import Path
import io

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ------------------------------------------------------------
# CIFAR-10 labels: indices must match the official dataset IDs.
# ------------------------------------------------------------
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
MODEL_PATH = Path(__file__).resolve().parent / "model.keras"
SAMPLE_COUNT_PER_CLASS = 1

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

@st.cache_resource(show_spinner="Loading the trained CIFAR-10 model...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH.name}. Keep model.keras in the same folder as app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_data(show_spinner="Downloading CIFAR-10 test images on first use...")
def load_cifar10_test_data():
    # The Keras CIFAR-10 test set contains real labelled examples.
    # Labels are retained from the dataset; no labels are manually assigned.
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return x_test, y_test.reshape(-1)

def image_for_model(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((32, 32), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def get_one_example_per_class(images, labels):
    """Return one image for each class, matched using its actual dataset label."""
    examples = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        matching_indices = np.flatnonzero(labels == class_id)
        if len(matching_indices) == 0:
            continue
        # Use the first actual occurrence of this class in the official test set.
        idx = int(matching_indices[0])
        examples[class_id] = (images[idx], int(labels[idx]), idx)
    return examples

def run_prediction(model, image):
    batch = image_for_model(image)
    probabilities = model.predict(batch, verbose=0)[0]
    predicted_id = int(np.argmax(probabilities))
    return predicted_id, probabilities

st.title("🖼️ CIFAR-10 Image Classification")
st.caption(
    "Explore all 10 CIFAR-10 categories, select a correctly labelled sample image, "
    "or upload your own image for CNN classification."
)

with st.expander("About the project and model inputs", expanded=True):
    st.write(
        "This application uses your supplied TensorFlow/Keras CNN model. "
        "CIFAR-10 contains 10 image categories. The model receives RGB images resized "
        "to 32 × 32 pixels with pixel values scaled to the 0–1 range."
    )
    st.info(
        "The sample gallery uses the official CIFAR-10 test labels. "
        "Each displayed sample name is derived from the label attached to that exact image."
    )

try:
    model = load_model()
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()

# ---------------------- Sample gallery ----------------------
st.header("1. CIFAR-10 sample gallery — all 10 classes")
st.write(
    "Click the button to load one real test image from each class. "
    "The first download may take a little time; subsequent loads are cached."
)

if st.button("Load all 10 correctly labelled sample images", type="primary"):
    try:
        x_test, y_test = load_cifar10_test_data()
        st.session_state["cifar10_examples"] = get_one_example_per_class(x_test, y_test)
    except Exception as exc:
        st.error(f"Could not load CIFAR-10 sample images: {exc}")

examples = st.session_state.get("cifar10_examples")

if examples:
    cols = st.columns(5)
    for class_id, class_name in enumerate(CLASS_NAMES):
        if class_id not in examples:
            continue
        img_array, actual_label_id, dataset_index = examples[class_id]
        # Display title is based on the image's actual label, not loop text.
        actual_label_name = CLASS_NAMES[actual_label_id]
        with cols[class_id % 5]:
            st.image(
                img_array,
                caption=f"Ground truth: {actual_label_name.title()}",
                use_container_width=True
            )
            st.caption(f"Dataset class ID: {actual_label_id}")
    st.caption(
        "Each caption is generated from the CIFAR-10 label stored alongside the displayed image."
    )
else:
    st.warning("Sample gallery has not been loaded yet. Click the button above to display all 10 classes.")

st.divider()

# ---------------------- Prediction testing ----------------------
st.header("2. Test the model")
input_mode = st.radio(
    "Choose an image source",
    ["Choose a sample from the CIFAR-10 gallery", "Upload your own image"],
    horizontal=False
)

selected_image = None
selected_ground_truth = None

if input_mode == "Choose a sample from the CIFAR-10 gallery":
    if not examples:
        st.info("First click **Load all 10 correctly labelled sample images** above.")
    else:
        selected_class_id = st.selectbox(
            "Select a sample class",
            list(examples.keys()),
            format_func=lambda class_id: (
                f"{CLASS_NAMES[int(examples[class_id][1])].title()} "
                f"(actual CIFAR-10 label)"
            )
        )
        img_array, actual_label_id, dataset_index = examples[selected_class_id]
        selected_image = Image.fromarray(img_array)
        selected_ground_truth = CLASS_NAMES[int(actual_label_id)]
        st.image(
            selected_image,
            caption=f"Selected image — actual label: {selected_ground_truth.title()}",
            width=260
        )
else:
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        try:
            selected_image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
            st.image(selected_image, caption="Uploaded image", width=260)
        except Exception as exc:
            st.error(f"Unable to open this image: {exc}")

if selected_image is not None:
    if st.button("Run prediction", type="primary"):
        try:
            with st.spinner("Classifying image..."):
                predicted_id, probabilities = run_prediction(model, selected_image)

            predicted_name = CLASS_NAMES[predicted_id]
            confidence = float(probabilities[predicted_id]) * 100

            st.subheader("Prediction result")
            st.success(f"Predicted class: **{predicted_name.title()}**")
            st.metric("Prediction confidence", f"{confidence:.2f}%")

            if selected_ground_truth is not None:
                st.write(f"**Ground-truth label:** {selected_ground_truth.title()}")
                if predicted_name == selected_ground_truth:
                    st.success("Correct prediction for this sample.")
                else:
                    st.warning("The model prediction differs from the dataset's ground-truth label.")

            st.write("Probability by class")
            probability_data = {
                CLASS_NAMES[i]: float(probabilities[i])
                for i in range(len(CLASS_NAMES))
            }
            st.bar_chart(probability_data)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.divider()
st.subheader("Input specifications")
st.markdown(
    """
    - **Accepted upload formats:** JPG, JPEG, PNG
    - **Colour channels:** RGB (grayscale uploads are converted to RGB)
    - **Model input shape:** 32 × 32 × 3
    - **Pixel scaling:** 0–255 converted to 0–1
    - **Output classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    """
)
st.caption(
    "Educational demonstration only. CIFAR-10-trained models are designed for small, "
    "low-resolution CIFAR-10-style images; predictions on unrelated real-world photographs may be unreliable."
)
