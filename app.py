import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# 1. CIFAR-10 Class labels
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# 2. Load Model (cached so it only loads once)
@st.cache_resource
def load_trained_model():
  return tf.keras.models.load_model("model.keras")


# 3. Cache CIFAR-10 test data for quick sample testing
@st.cache_data
def get_sample_data():
  (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  return x_test, y_test


model = load_trained_model()
x_test, y_test = get_sample_data()

st.title("CIFAR-10 Image Classifier")

# Choose between uploading or testing with built-in CIFAR-10 images
test_mode = st.radio(
    "Choose Input Method:", ("Upload an Image", "Use a Built-in Test Sample")
)

target_image = None
true_label = None

if test_mode == "Upload an Image":
  uploaded_file = st.file_uploader(
      "Choose an image...", type=["jpg", "jpeg", "png"]
  )
  if uploaded_file is not None:
    target_image = Image.open(uploaded_file).convert("RGB")
    target_image = target_image.resize((32, 32))

else:
  st.subheader("Select a Test Image from CIFAR-10")
  # Pre-selected indices for Airplane (0), Ship (8), and Truck (9)
  sample_choices = {
      "Sample 1: Airplane": 0,
      "Sample 2: Ship": 8,
      "Sample 3: Truck": 9,
  }
  selected_choice = st.selectbox(
      "Pick a sample image:", list(sample_choices.keys())
  )

  idx = sample_choices[selected_choice]
  target_image = Image.fromarray(x_test[idx]).resize((32, 32))
  true_label = CLASS_NAMES[y_test[idx][0]]

# Display and predict if an image is selected/loaded
if target_image is not None:
  # Display the 32x32 image (enlarged for visibility in the browser)
  st.image(
      target_image,
      caption="Input Image (Resized to 32x32)",
      width=200,
  )

  if true_label:
    st.info(f"Ground Truth Label: **{true_label}**")

  if st.button("Run Prediction"):
    # Preprocessing
    img_array = np.array(target_image, dtype=np.float32)

    # Normalize if your model was trained with normalized inputs (0 to 1)
    img_array = img_array / 255.0

    # Add batch dimension: (32, 32, 3) -> (1, 32, 32, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction[0])
    confidence = float(np.max(prediction[0])) * 100

    predicted_label = CLASS_NAMES[predicted_class_idx]

    st.success(
        f"**Predicted Class:** {predicted_label} ({confidence:.2f}% confidence)"
    )