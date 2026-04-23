from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.60
MODEL_CANDIDATES = [
    APP_DIR / "tomato_efficientnetb0.keras",
    APP_DIR / "tomato_efficientnetb0.h5",
    APP_DIR / "models" / "tomato_efficientnetb0.keras",
    APP_DIR / "models" / "tomato_efficientnetb0.h5",
]
METRICS_PATH = APP_DIR / "model_accuracy_metrics.csv"
CONFUSION_MATRIX_IMAGE_CANDIDATES = [
    APP_DIR / "confusion_matrix_efficientnetb0.png",
    APP_DIR / "Confusion matrix1.png",
]

CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "name": "Bacterial Spot",
        "description": "Small dark, water-soaked lesions that can expand on leaves and fruit. It is commonly favored by warm, wet conditions.",
        "treatment": "Remove infected debris, avoid overhead watering, improve spacing, and use copper-based bactericides where locally recommended.",
    },
    "Tomato___Early_blight": {
        "name": "Early Blight",
        "description": "A fungal disease that often causes brown spots with concentric rings, usually starting on older lower leaves.",
        "treatment": "Prune affected leaves, mulch to reduce soil splash, rotate crops, and apply a suitable fungicide such as chlorothalonil or copper.",
    },
    "Tomato___Late_blight": {
        "name": "Late Blight",
        "description": "A fast-spreading disease that creates irregular brown patches, often with pale mold under humid conditions.",
        "treatment": "Remove badly infected plants quickly, avoid wet foliage, and use preventive fungicides such as mancozeb or metalaxyl where appropriate.",
    },
    "Tomato___Leaf_Mold": {
        "name": "Leaf Mold",
        "description": "Yellow patches on upper leaf surfaces with olive-gray mold on the underside, especially in humid greenhouses.",
        "treatment": "Increase airflow, reduce humidity, avoid overcrowding, and apply labeled fungicides if disease pressure is high.",
    },
    "Tomato___Septoria_leaf_spot": {
        "name": "Septoria Leaf Spot",
        "description": "Many small circular spots with dark borders, usually appearing first on lower leaves.",
        "treatment": "Remove infected leaves, keep foliage dry, clear crop residue, and use protective fungicides when needed.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name": "Two-spotted Spider Mite",
        "description": "Tiny mites cause yellow stippling, bronzing, and sometimes fine webbing on leaves.",
        "treatment": "Spray leaf undersides with water, use insecticidal soap or neem oil, and avoid dusty, dry growing conditions.",
    },
    "Tomato___Target_Spot": {
        "name": "Target Spot",
        "description": "Circular brown lesions with ring-like patterns that can affect leaves, stems, and fruit.",
        "treatment": "Remove infected material, improve airflow, avoid overhead irrigation, and use preventive fungicides.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "description": "A viral disease spread mainly by whiteflies, causing yellowing, curling leaves, and stunted growth.",
        "treatment": "Control whiteflies, remove infected plants, manage weeds, and plant resistant varieties when possible.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "description": "A viral disease that can cause mottled, curled, or distorted leaves and reduced plant vigor.",
        "treatment": "Use virus-free seed, disinfect tools, wash hands after handling plants, and remove infected plants.",
    },
    "Tomato___healthy": {
        "name": "Healthy Tomato Leaf",
        "description": "The leaf does not show strong visual signs of the diseases represented in the trained classes.",
        "treatment": "Keep monitoring, water at soil level, maintain balanced nutrition, and inspect plants weekly.",
    },
}


def find_model_path() -> Path | None:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    return None


@st.cache_resource(show_spinner="Loading EfficientNetB0 classifier...")
def load_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


def load_class_names() -> list[str]:
    npy_path = APP_DIR / "class_names.npy"
    if npy_path.exists():
        names = np.load(npy_path, allow_pickle=True).tolist()
        return [str(name) for name in names]
    return CLASS_NAMES


@st.cache_data(show_spinner=False)
def load_model_metrics() -> dict[str, float]:
    if not METRICS_PATH.exists():
        return {}
    metrics = pd.read_csv(METRICS_PATH)
    if metrics.empty:
        return {}
    return {
        column: float(metrics.loc[0, column])
        for column in metrics.columns
        if pd.notna(metrics.loc[0, column])
    }


def find_confusion_matrix_image() -> Path | None:
    for path in CONFUSION_MATRIX_IMAGE_CANDIDATES:
        if path.exists():
            return path
    return None


def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def estimate_leaf_score(image: Image.Image) -> float:
    """Lightweight fallback validator for obvious non-leaf uploads."""
    rgb = np.asarray(image.convert("RGB").resize(IMAGE_SIZE), dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    green_mask = cv2.inRange(hsv, (25, 35, 30), (95, 255, 255))
    yellow_brown_mask = cv2.inRange(hsv, (8, 30, 25), (45, 255, 230))
    vegetation_ratio = float(np.count_nonzero(green_mask | yellow_brown_mask)) / green_mask.size

    saturation_score = float(np.mean(hsv[:, :, 1] > 35))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edge_density = float(np.count_nonzero(edges)) / edges.size
    edge_score = min(edge_density / 0.12, 1.0)

    return 0.65 * vegetation_ratio + 0.20 * saturation_score + 0.15 * edge_score


def is_probably_tomato_leaf(predicted_class: str, confidence: float, leaf_score: float) -> bool:
    is_tomato_class = predicted_class.startswith("Tomato___")
    return bool(is_tomato_class and (confidence >= 0.35 or leaf_score >= 0.45))


def predict(model: tf.keras.Model, image: Image.Image, class_names: list[str]) -> tuple[str, float, np.ndarray]:
    batch = prepare_image(image)
    probabilities = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    return class_names[index], float(probabilities[index]), probabilities


def get_disease_info(predicted_class: str) -> dict[str, str]:
    normalized_class = predicted_class.replace("_Two-spotted_", " Two-spotted_")
    fallback_name = predicted_class.replace("Tomato___", "").replace("_", " ")
    return DISEASE_INFO.get(
        predicted_class,
        DISEASE_INFO.get(
            normalized_class,
            {
                "name": fallback_name,
                "description": "No description is available for this class.",
                "treatment": "Consult a local agriculture expert for confirmation and treatment guidance.",
            },
        ),
    )


def clean_class_name(class_name: str) -> str:
    return class_name.replace("Tomato___", "").replace("_", " ")


def build_probability_frame(class_names: list[str], probabilities: np.ndarray, predicted_class: str) -> pd.DataFrame:
    rows = []
    for class_name, probability in zip(class_names, probabilities):
        label = clean_class_name(class_name)
        if class_name == predicted_class:
            label = f"{label} (predicted)"
        rows.append({"Category": label, "Probability": float(probability) * 100})
    return pd.DataFrame(rows).sort_values("Probability", ascending=False)


def plot_probability_chart(class_names: list[str], probabilities: np.ndarray):
    labels = [name.replace("_", "__") for name in class_names]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(labels, probabilities, color="green")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Disease Class")
    ax.set_xlim(0, max(float(np.max(probabilities)) * 1.15, 1.0))
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


FARMING_TIPS = [
    "Rotate tomato crops every 2 years to reduce soil-borne diseases.",
    "Avoid overhead watering to reduce fungal infections.",
    "Use neem oil as a natural pesticide for mite control.",
    "Check leaves weekly so diseases are detected early.",
    "Use disease-resistant tomato varieties whenever possible.",
]


st.set_page_config(page_title="Tomato Leaf Disease Detection", page_icon="🍅", layout="centered")

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
        .section-title {
            font-size: 1.45rem;
            font-weight: 700;
            margin: 1.35rem 0 0.65rem;
            color: #ffffff;
        }
        .compact-text {
            font-size: 0.95rem;
            font-weight: 600;
            color: #ffffff;
        }
        .tip-box {
            background: #0f3d24;
            color: #42ff75;
            padding: 0.9rem;
            border-radius: 0.35rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Tomato Leaf Disease Detection")
st.caption("EfficientNetB0 classifier with model accuracy, confidence, class probabilities, confusion matrix, and remedy output.")

with st.sidebar:
    st.header("Model")
    st.write("Input size: 224 x 224")
    st.write(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    st.write("Architecture: EfficientNetB0 + custom softmax head")

model_path = find_model_path()
class_names = load_class_names()
metrics = load_model_metrics()
confusion_matrix_image = find_confusion_matrix_image()

if model_path is None:
    st.warning(
        "No trained EfficientNetB0 model was found. Train the notebook first so it saves "
        "`tomato_efficientnetb0.keras`, then rerun this app."
    )
    st.stop()

model = load_model(str(model_path))
uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is None:
    st.info("Upload an image to begin.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded image", use_container_width=True)

leaf_score = estimate_leaf_score(image)
if leaf_score < 0.18:
    st.error("Invalid input: Please upload an image of a leaf.")
    st.stop()

predicted_class, confidence, probabilities = predict(model, image, class_names)
probability_frame = build_probability_frame(class_names, probabilities, predicted_class)

if not is_probably_tomato_leaf(predicted_class, confidence, leaf_score):
    st.error("Invalid input: The image does not appear to be a tomato leaf.")
    st.stop()

info = get_disease_info(predicted_class)

display_class = predicted_class.replace("___", "___").replace("_", "__")
st.success(f"Predicted Class: {display_class}")
st.progress(min(int(confidence * 100), 100))
st.info(f"Model Confidence: {confidence * 100:.2f}%")

if "Accuracy" in metrics:
    st.info(f"Model Accuracy: {metrics['Accuracy'] * 100:.2f}%")

if confidence < CONFIDENCE_THRESHOLD:
    st.warning("Low confidence prediction. Please upload a clearer image.")

st.markdown('<div class="section-title">Disease Information</div>', unsafe_allow_html=True)
st.markdown(f'<div class="compact-text">Plant Class: Tomato leaf</div>', unsafe_allow_html=True)
st.markdown(f'<div class="compact-text">Disease Category: {info["name"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="compact-text">Description: {info["description"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="compact-text">Suggested Solution: {info["treatment"]}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Confidence by Disease Class</div>', unsafe_allow_html=True)
fig = plot_probability_chart(class_names, probabilities)
st.pyplot(fig)

report = f"""Tomato Leaf Disease Detection Report
------------------------------------
Plant Class: Tomato leaf
Predicted Class: {predicted_class}
Disease Name: {info['name']}
Confidence: {confidence * 100:.2f}%
Model Accuracy: {metrics.get('Accuracy', 0.0) * 100:.2f}%

Description:
{info['description']}

Suggested Solution:
{info['treatment']}
"""

st.download_button(
    "Download Diagnosis Report",
    report,
    file_name="tomato_leaf_diagnosis_report.txt",
    mime="text/plain",
)

st.markdown('<div class="section-title">Pro Farming Tip</div>', unsafe_allow_html=True)
st.markdown(f'<div class="tip-box">{random.choice(FARMING_TIPS)}</div>', unsafe_allow_html=True)

st.subheader("Confusion Matrix")
if confusion_matrix_image is not None:
    st.image(str(confusion_matrix_image), caption="Confusion matrix from notebook evaluation", use_container_width=True)
else:
    st.info("Confusion matrix will appear after running the notebook evaluation cell.")

with st.expander("Exact class probabilities"):
    display_frame = probability_frame.copy()
    display_frame["Probability"] = display_frame["Probability"].map(lambda value: f"{value:.2f}%")
    st.dataframe(display_frame, hide_index=True, use_container_width=True)
