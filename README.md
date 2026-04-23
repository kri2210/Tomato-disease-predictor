# Tomato Disease Prediction

This project is a tomato leaf disease classification app built with EfficientNetB0 transfer learning and a Streamlit interface. It predicts the leaf class, shows confidence scores, displays disease information and treatment guidance, and includes a downloadable diagnosis report.

## Features

- Tomato leaf disease prediction with the trained `tomato_efficientnetb0.keras` model.
- Confidence display, probability chart, and model accuracy summary.
- Basic input validation to reject obvious non-leaf images.
- Disease descriptions and suggested treatments for each supported class.
- Confusion matrix preview and a downloadable text report.

## Project Files

- `app.py` - Streamlit inference app.
- `EfficientNetB0_Tomato_Disease_Detection.ipynb` - training and evaluation notebook.
- `tomato_efficientnetb0.keras` - trained model used by the app.
- `class_names.npy` - saved class label order.
- `model_accuracy_metrics.csv` - accuracy and metric summary.
- `confusion_matrix_values.csv` - confusion matrix values used for analysis.
- `requirements.txt` - Python dependencies.

## Local Setup

1. Create and activate a Python environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

The app looks for the model in the project folder first. If it cannot find a trained model, run the notebook to generate `tomato_efficientnetb0.keras` and `class_names.npy`.

## Training Notebook

Open `EfficientNetB0_Tomato_Disease_Detection.ipynb` and run the cells to train and evaluate the model.

The notebook is expected to save the following outputs:

- `tomato_efficientnetb0.keras`
- `class_names.npy`
- `confusion_matrix_efficientnetb0.png`

## Dataset Layout

The notebook expects the tomato dataset to be arranged with separate class folders, such as:

```text
Dataset/
	train/
	valid/
```

If your dataset uses different folder names, update the notebook paths before training.

## Deploy On GitHub

This project is ready to be published to a GitHub repository and deployed from that repository.
## Streamlit deployed link
https://tomato-disease-predictor-22.streamlit.app/
