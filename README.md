# CIFAR-10 Image Classification

A TensorFlow CNN that classifies 32×32 RGB images into the 10 CIFAR-10 categories, with a Streamlit demo and a separate FastAPI service for deployment.

## Streamlit
Install dependencies with `pip install -r requirements.txt`, then run `streamlit run app.py`.

## Render API
Set the Render root directory to the repository root, build command to `pip install -r render_api/requirements.txt`, and start command to `uvicorn render_api.main:app --host 0.0.0.0 --port $PORT`. The API exposes `/health`, `/docs`, and `POST /predict` (multipart image field named `file`).

## Files
- `model.keras`: supplied trained Keras model
- `app.py`: Streamlit interface with image upload and on-demand CIFAR-10 examples
- `render_api/main.py`: FastAPI prediction endpoint
- `Image_Visualization_CIFAR10_Deployable.ipynb`: original analysis notebook with a minimal syntax correction and deployment note

## Important
The CIFAR-10 sample set is downloaded by Keras only when a user explicitly clicks **Load CIFAR-10 sample images**. This avoids an automatic large download during initial app startup. Uploading an image works without downloading the CIFAR-10 dataset.
