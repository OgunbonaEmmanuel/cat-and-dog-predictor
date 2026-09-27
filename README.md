# Cat and Dog Predictor

An image classification project with a TensorFlow/Keras convolutional neural network, a FastAPI prediction service, and a Streamlit interface. Upload a JPG, JPEG, or PNG image to receive a Cat or Dog prediction.

The repository also includes a model-building notebook with training output and visualizations of convolutional feature maps and max-pooling outputs.

## Features

- Image upload and preview in Streamlit.
- Predictions through a FastAPI `POST /predict` endpoint.
- A custom CNN trained on the TensorFlow Datasets `cats_vs_dogs` dataset.
- Saved Keras model loading once when the backend starts.
- Training and intermediate activation visualizations in the notebook.

## Project structure

| Path | Purpose |
| --- | --- |
| `backend/main.py` | FastAPI prediction service |
| `backend/cats_vs_dogs_classifier.keras` | Saved classifier |
| `frontend/main.py` | Streamlit interface |
| `cats_vs_dogs_model_build.ipynb` | Training and feature-map exploration |
| `PracImages/` | Practice images, grouped into Cat, CatAndDog, and dog folders |
| `requirements.txt` | App and notebook dependencies |
| `.gitignore` | Local environment and cache exclusions |

The classifier returns one label for the whole image. A `CatAndDog` practice folder does not imply the model can detect both animals independently.

## Model and training

The notebook loads the first 80% of the dataset's training split for training and the remaining 20% for validation. Although the second variable is named `test_set`, it is passed to `model.fit()` as validation data; it is not a separate final test set.

Images are resized to 150 × 150 and divided by 255. Batches contain 32 images, with prefetching enabled.

| Layer | Configuration |
| --- | --- |
| Input | 150 × 150 × 3 |
| Convolution | 32 filters, 3 × 3 kernel, ReLU |
| Max pooling | 2 × 2 |
| Convolution | 64 filters, 3 × 3 kernel, ReLU |
| Max pooling | 2 × 2 |
| Flatten | Converts feature maps to a vector |
| Dense | 64 units, ReLU |
| Output | 1 unit, sigmoid |

Training uses Adam, binary cross-entropy, and five epochs. The notebook records Cat as class 0 and Dog as class 1.

### Recorded results

The uploaded notebook's saved training output reports:

| Metric | Value |
| --- | --- |
| Best validation accuracy | 76.98% at epoch 4 |
| Final validation accuracy | 76.20% at epoch 5 |
| Final validation loss | 0.7859 |

The export occurs after epoch 5; the notebook does not restore the best epoch. Rising validation loss while training loss falls suggests overfitting. These are recorded notebook results, not a new independent evaluation, and the supplied model binary has not been checked against this run.

## Installation

From the project root, create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Use a Python version supported by the TensorFlow release you install. Versions are unpinned because a tested application environment was not supplied. Use a compatible TensorFlow/Keras environment for model loading.

## Run the application

Keep the saved model at `backend/cats_vs_dogs_classifier.keras`.

### Terminal 1: backend

Activate the environment, then run from the project root:

```bash
cd backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Start the backend from this folder because the model-loading path is relative to the working directory. The local interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### Terminal 2: frontend

Open another terminal at the project root, activate the same environment, and run:

```bash
python -m streamlit run frontend/main.py
```

Open the address shown by Streamlit and upload an image. The frontend previews it and immediately sends it to the backend; there is no separate prediction button.

The application uses a local model and requires no API key.

## Prediction endpoint

`POST /predict` accepts a multipart upload under the field name `file`.

The backend resizes the image to 150 × 150, divides pixel values by 255, adds a batch dimension, and runs inference. An output greater than 0.5 returns Dog; otherwise it returns Cat.

Example response:

```json
{
  "prediction": "Cat",
  "confidence": 0.12
}
```

In the supplied code, `confidence` is always the raw sigmoid score for the Dog class. In this example, the corresponding Cat score is 0.88. It is not currently the confidence of the returned label, and neither score has been shown to be calibrated.

For label-specific scores, change the returned confidence to `probability if label == "Dog" else 1 - probability`, or rename the existing field to `dog_probability`.

## Rebuild the model

Open `cats_vs_dogs_model_build.ipynb` in VS Code or launch Jupyter with:

```bash
python -m notebook
```

Run the notebook in order using the project environment. TensorFlow Datasets downloads and prepares `cats_vs_dogs`; the notebook's recorded metadata reports a download of about 787 MiB, so allow sufficient disk space and time.

The export cell saves `cats_vs_dogs_classifier.keras` in the notebook's working directory. Copy or move that export into `backend/` before starting the API.

The recorded dataset metadata attributes the dataset to the Microsoft/Asirra cats-and-dogs work: Elson, Douceur, Howell, and Saul, *Asirra: A CAPTCHA that Exploits Interest-Aligned Manual Image Categorization*, ACM CCS, 2007. Review the dataset's terms before redistributing images.

## Known limitations and improvements

- The backend does not convert uploads to RGB. To support grayscale and transparent images, change image loading to `Image.open(io.BytesIO(image_bytes)).convert("RGB")`.
- The backend uses Pillow resizing while training uses TensorFlow resizing; align interpolation for more consistent preprocessing.
- Invalid images and excessive upload sizes are not explicitly handled.
- The frontend has no request timeout or handling for backend connection errors and non-JSON responses.
- Streamlit reruns may submit the same uploaded image again.
- Images containing both animals or neither animal still receive one of the two labels.
- No data augmentation, dropout, early stopping, or held-out final evaluation is implemented in the supplied notebook.
- The frontend backend URL is hardcoded to localhost; deployment would require appropriate service configuration.

The source and notebook were reviewed, but training and end-to-end inference were not rerun during documentation preparation.

## GitHub preparation

The accompanying `.gitignore` excludes Python caches, virtual environments, and secrets. It keeps the model and practice images eligible for upload. Check model and image file sizes before committing, and include only practice images you are permitted to share.
