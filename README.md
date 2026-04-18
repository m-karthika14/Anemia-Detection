# AI-Based Anemia Detection

Detects anemia from conjunctiva (eye) and fingernail images using handcrafted color/edge features and classical ML models.

**Project**
This project demonstrates a lightweight, explainable pipeline for screening anemia using two complementary visual cues: the conjunctiva (eye) and the fingernail. It focuses on reproducible preprocessing, handcrafted color and edge features, and classical ML classifiers (Random Forest / SVM / Naive Bayes) rather than end-to-end deep models, enabling fast training and simple interpretability. The repository contains scripts for preparing data, training models, evaluating performance, and a Streamlit demo for easy experimentation.

**Pipeline (high level)**
1. Data acquisition & preparation — obtain eye and nail datasets (e.g., via Kaggle) and run `setup_data.py` to extract and structure the files. Use `create_eye_labels.py` and `create_nail_pseudo_labels.py` to generate required CSVs.
2. Preprocessing & segmentation — images are preprocessed and segmented using functions in `utils/image_processing.py` (resizing, masking, edge detection).
3. Feature extraction — handcrafted color (HSV/chromatic) and edge features are computed in `utils/feature_extraction.py` for each image/mask pair.
4. Training — `train_eye.py` and `train_nail.py` train scikit-learn models and save them to `models/` (default outputs `eye_model.pkl`, `nail_model.pkl`).
5. Model comparison & selection — `compare_models.py` evaluates multiple classifiers on the eye dataset and writes `outputs/model_comparison.csv` and an accuracy plot used by the demo.
6. Prediction & fusion — use `predict.py` or the Streamlit app (`app.py`) to score images. The demo fuses eye/nail model probabilities using weights (defaults in `app.py`: `EYE_WEIGHT=0.7`, `NAIL_WEIGHT=0.3`) and a threshold (`THRESHOLD=0.5`) to produce a final label.
7. Evaluation & visualization — `evaluate.py` computes summary metrics (precision/recall/f1) and saves visual examples to `outputs/`.


**Repository layout (important files)**
- **[app.py](app.py)**: Streamlit-based demo UI for uploading images and getting predictions.
- **[predict.py](predict.py)**: CLI tool to score a pair of eye/nail images and print a label.
- **[main.py](main.py)**: High-level entry script to run labels, training, prediction, and evaluation modes.
- **[train_eye.py](train_eye.py)**, **[train_nail.py](train_nail.py)**: Training scripts for eye and nail models.
- **[setup_data.py](setup_data.py)**: Helper to extract/prepare datasets downloaded externally.
- **[utils/](utils/)**: Data loading, feature extraction and image-processing helpers.

**Supported OS / Python**
- Windows / Linux / macOS
- Python 3.8+ recommended (works with Python 3.8–3.11)

**Dependencies**
All Python deps are listed in `requirements.txt`. Key packages:
- **numpy**, **opencv-python**, **scikit-image**, **scikit-learn==1.8.0**, **pandas**, **joblib**, **matplotlib**, **seaborn**, **tqdm**, **streamlit**, **openpyxl**

Install:

```bash
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
# bash / macOS / WSL
source .venv/bin/activate

pip install -r requirements.txt
```

**Data**
- This repository does not include the raw datasets. Use `setup_data.py` to unpack/prep datasets you obtain (example datasets are available on Kaggle).
- Expected basic layout after `setup_data.py` / manual organization:
	- `data/eye/` — images, masks, and `meta/labels.csv` for eye conjunctiva
	- `data/nail/images/` and `data/nail/masks/` — nail images and segmentation masks

Example (Kaggle) workflow:

```bash
# download datasets manually or via kaggle CLI
python setup_data.py --eye-zip path/to/anemia-dataset.zip --nail-zip path/to/nail-dataset.zip --out-dir data
```

Refer to the original dataset pages for licensing and terms before using them.

**Quick Usage**

- Run demo Streamlit app (recommended for quick checks):

```bash
streamlit run app.py
```

This opens a web UI where you can upload an eye image (and optional mask) and a nail image (and optional mask). The app loads models from `models/eye_model.pkl` and `models/nail_model.pkl` (or uses a recommended eye model chosen from `outputs/model_comparison.csv`).

- Run prediction from the command line:

```bash
python predict.py --eye_image PATH/TO/EYE.jpg --eye_mask PATH/TO/EYE_MASK.png \
		--nail_image PATH/TO/NAIL.jpg --nail_mask PATH/TO/NAIL_MASK.png
```

`predict.py` prints `eye_score`, `nail_score`, `final_score` and a final label (`Anemia Detected` / `Normal`). At least one of the eye or nail image+mask pairs must be provided.

- High-level combined workflow using `main.py`:

```bash
# run everything (labels -> train -> predict -> evaluate)
python main.py --mode all

# run only training
python main.py --mode train

# run only prediction (you can pass --eye_image, --eye_mask, --nail_image, --nail_mask)
python main.py --mode predict --eye_image data/eye/images/.. --eye_mask data/eye/masks/..
```

**Training**
- Train the eye-model (example):

```bash
python train_eye.py --data data/eye --out models/eye_model.pkl
```

- Train the nail-model (if you have labels/masks):

```bash
python train_nail.py --images data/nail/images --labels data/nail/labels_pseudo.csv --out models/nail_model.pkl
```

After training, models are saved to the `models/` directory by default.

**Evaluation and model comparison**
- Use `compare_models.py` to evaluate different classifiers on the eye dataset. It writes `outputs/model_comparison.csv` and a plot `outputs/model_accuracy_comparison.png` used by the Streamlit UI.
- Use `evaluate.py` to compute precision/recall/f1 and visualize samples. Example:

```bash
python evaluate.py --eye_model models/eye_model.pkl --labels_csv data/eye/meta/labels.csv
```

**Outputs**
- `models/` — trained model files (e.g., `eye_model.pkl`, `nail_model.pkl`)
- `outputs/` — evaluation outputs, plots, comparison CSVs, and predictions

**Troubleshooting**
- Model fails to load: ensure model files exist in `models/` and were created with compatible scikit-learn/joblib versions.
- OpenCV image decode errors: verify uploaded images are valid JPEG/PNG and masks are grayscale or binary. The Streamlit UI decodes files using OpenCV.
- If `predict.py` complains about missing arguments: provide at least one complete image+mask pair (`--eye_image`+`--eye_mask` or `--nail_image`+`--nail_mask`).

**Development notes**
- Feature extraction is implemented in `utils/feature_extraction.py` and preprocessing in `utils/image_processing.py`.
- The Streamlit demo combines eye/nail model scores with weights defined in `app.py` (`EYE_WEIGHT`, `NAIL_WEIGHT`, and `THRESHOLD`). Adjust these values if you retrain models and want a different fusion.

**Reproducibility**
- Use the provided scripts (`create_eye_labels.py`, `create_nail_pseudo_labels.py`) to generate label CSVs used by training and evaluation.

**Contributing / License**
- This repo is a research/demo scaffold. If you plan to reuse datasets, check dataset licenses before redistribution.
- Open a GitHub issue or PR with reproducible steps for fixes or improvements.



