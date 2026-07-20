"""SafeVison Streamlit dashboard for image-based human fall detection."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_PATH = APP_DIR / "model" / "final_resnet50.keras"
IMAGE_SIZE = (224, 224)
DECISION_THRESHOLD = 0.5
PRIMARY_PAGES = [
    "Home",
    "Live Prediction",
    "Project Workflow",
]
WORKFLOW_PAGES = [
    "Workflow Overview",
    "Dataset & EDA",
    "Model Architectures",
    "Training & Hyperparameters",
    "Model Comparison",
    "Final Model Evaluation",
    "Calibration",
    "Error Analysis",
    "Grad-CAM Explainability",
    "Blockers & Solutions",
    "Technology Stack",
    "Limitations & Future Scope",
]

st.set_page_config(
    page_title="SafeVison | Fall Detection",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading ResNet50 inference model...")
def load_model() -> tf.keras.Model:
    """Load the deployment model once without restoring training state."""
    if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"Model artifact is missing: {MODEL_PATH.name}")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data(show_spinner=False)
def load_csv_safely(relative_path: str) -> pd.DataFrame | None:
    """Load a project CSV, returning None when unavailable or invalid."""
    path = PROJECT_ROOT / relative_path
    try:
        return pd.read_csv(path) if path.is_file() else None
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None


@st.cache_data(show_spinner=False)
def load_history(relative_path: str) -> pd.DataFrame | None:
    """Load and normalize a saved Keras CSV history."""
    frame = load_csv_safely(relative_path)
    if frame is None or frame.empty:
        return None
    if "epoch" not in frame.columns:
        frame = frame.copy()
        frame.insert(0, "epoch", np.arange(1, len(frame) + 1))
    return frame


def load_image_safely(path: Path) -> Image.Image | None:
    """Open an image artifact without taking down the current page."""
    try:
        with Image.open(path) as image:
            return image.convert("RGB").copy()
    except (FileNotFoundError, OSError, UnidentifiedImageError):
        return None


def preprocess_image(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """Prepare RGB pixels; ResNet preprocessing is embedded in the saved model."""
    display_image = ImageOps.exif_transpose(image).convert("RGB")
    resized = display_image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    return display_image, batch


def predict_image(model: tf.keras.Model, batch: np.ndarray) -> float:
    """Return the clipped sigmoid output, verified by evaluation as P(Fall)."""
    output = np.asarray(model.predict(batch, verbose=0)).squeeze()
    if output.size != 1:
        raise ValueError(f"Expected one output value; received shape {output.shape}.")
    return float(np.clip(output.item(), 0.0, 1.0))


def apply_theme(theme: str) -> None:
    """Apply a light or dark token set to Streamlit and custom components."""
    dark = theme == "Dark"
    colors = {
        "bg": "#0b1118" if dark else "#f5f7fa",
        "surface": "#121b25" if dark else "#ffffff",
        "surface2": "#182431" if dark else "#eef2f6",
        "text": "#edf3f8" if dark else "#17212b",
        "muted": "#9fb0c0" if dark else "#5f6d79",
        "border": "#2a3a49" if dark else "#d9e0e7",
        "accent": "#35a7d7" if dark else "#087ea4",
        "accent2": "#e0a43a" if dark else "#a96508",
        "success": "#46b889" if dark else "#167a55",
        "danger": "#ef6b73" if dark else "#be3341",
        "shadow": "none" if dark else "0 8px 24px rgba(27, 39, 51, .06)",
    }
    st.markdown(
        f"""
        <style>
        :root {{
            --app-bg:{colors['bg']}; --surface:{colors['surface']};
            --surface-2:{colors['surface2']}; --text:{colors['text']};
            --muted:{colors['muted']}; --border:{colors['border']};
            --accent:{colors['accent']}; --accent-2:{colors['accent2']};
            --success:{colors['success']}; --danger:{colors['danger']};
            --shadow:{colors['shadow']};
        }}
        .stApp, [data-testid="stAppViewContainer"] {{ background:var(--app-bg); color:var(--text); }}
        [data-testid="stHeader"] {{ background:rgba(0,0,0,0); }}
        [data-testid="stSidebar"] {{ background:var(--surface); border-right:1px solid var(--border); }}
        [data-testid="stSidebarContent"] {{ padding:1.25rem .85rem; }}
        .sidebar-brand {{
            padding:1rem; margin:0 0 1rem; border:1px solid var(--border);
            border-radius:12px; background:linear-gradient(145deg, var(--surface-2), var(--surface));
            box-shadow:var(--shadow);
        }}
        .sidebar-brand-icon {{
            width:2.25rem; height:2.25rem; display:flex; align-items:center; justify-content:center;
            margin-bottom:.75rem; border-radius:9px; background:var(--accent); color:#fff; font-size:1.2rem;
        }}
        .sidebar-brand-title {{ color:var(--text); font-size:1.05rem; font-weight:750; line-height:1.25; }}
        .sidebar-brand-copy {{ color:var(--muted); font-size:.78rem; margin-top:.3rem; }}
        .nav-section-label {{
            color:var(--muted); font-size:.68rem; font-weight:750; letter-spacing:.11em;
            text-transform:uppercase; margin:.35rem .5rem .25rem;
        }}
        .workflow-nav {{
            border-left:2px solid var(--accent); margin:.15rem 0 .75rem .65rem;
            padding:.55rem .7rem; background:var(--surface-2); border-radius:0 10px 10px 0;
            color:var(--accent); font-size:.7rem; font-weight:750; letter-spacing:.08em;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label {{
            padding:.42rem .55rem; border-radius:7px; transition:background .15s ease, color .15s ease;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background:var(--surface-2); }}
        [data-testid="stSidebar"] hr {{ margin:.85rem 0; }}
        .model-badge {{
            padding:.7rem .8rem; border:1px solid var(--border); border-radius:9px;
            background:var(--surface-2); color:var(--muted); font-size:.76rem; line-height:1.55;
        }}
        .model-badge strong {{ color:var(--text); font-size:.8rem; }}
        [data-testid="stMainBlockContainer"] {{ max-width:1280px; padding-top:2rem; padding-bottom:4rem; }}
        h1,h2,h3,h4,p,label,[data-testid="stMarkdownContainer"] {{ color:var(--text); letter-spacing:0; }}
        .muted, .muted p {{ color:var(--muted) !important; }}
        .page-kicker {{ color:var(--accent); font-size:.78rem; font-weight:700; text-transform:uppercase; margin-bottom:.45rem; }}
        .page-title {{ font-size:2.25rem; line-height:1.12; font-weight:750; color:var(--text); margin:0 0 .55rem; }}
        .page-copy {{ max-width:820px; color:var(--muted); font-size:1rem; line-height:1.65; margin-bottom:1.6rem; }}
        .panel {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:1.15rem; box-shadow:var(--shadow); height:100%; }}
        .panel-title {{ color:var(--text); font-weight:700; font-size:1rem; margin-bottom:.4rem; }}
        .panel-copy {{ color:var(--muted); font-size:.9rem; line-height:1.55; }}
        .pipeline {{ display:flex; flex-wrap:wrap; gap:.45rem; align-items:center; margin:1rem 0 1.8rem; }}
        .pipeline-step {{ background:var(--surface); border:1px solid var(--border); color:var(--text); padding:.52rem .7rem; border-radius:6px; font-size:.82rem; }}
        .pipeline-arrow {{ color:var(--accent); font-weight:700; }}
        .status-line {{ border-left:3px solid var(--accent); background:var(--surface); padding:.8rem 1rem; color:var(--muted); margin:1rem 0; }}
        [data-testid="stMetric"] {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:1rem; box-shadow:var(--shadow); }}
        [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {{ color:var(--muted); }}
        [data-testid="stMetricValue"] {{ color:var(--text); }}
        [data-testid="stFileUploaderDropzone"], [data-testid="stCameraInput"] {{ background:var(--surface); border-color:var(--border); border-radius:8px; }}
        [data-testid="stExpander"], [data-testid="stDataFrame"], div[data-baseweb="tab-list"] {{ background:var(--surface); border-color:var(--border); }}
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{ background:var(--surface); border-color:var(--border); color:var(--text); }}
        .stButton > button, .stDownloadButton > button {{ border-radius:6px; border:1px solid var(--border); background:var(--surface-2); color:var(--text); }}
        .stButton > button[kind="primary"] {{ background:var(--accent); color:#ffffff; border-color:var(--accent); }}
        hr {{ border-color:var(--border); }}
        @media (max-width:700px) {{ .page-title {{ font-size:1.75rem; }} [data-testid="stMainBlockContainer"] {{ padding-top:1.2rem; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render persistent project identity, theme control, and page navigation."""
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand"><div class="sidebar-brand-icon">&#10010;</div>'
            '<div class="sidebar-brand-title">SafeVison</div>'
            '<div class="sidebar-brand-copy">AI-powered fall detection</div></div>',
            unsafe_allow_html=True,
        )
        if "theme" not in st.session_state:
            st.session_state.theme = "Dark"
        theme = st.toggle("Dark mode", value=st.session_state.theme == "Dark")
        st.session_state.theme = "Dark" if theme else "Light"
        st.divider()

        st.markdown('<div class="nav-section-label">Navigate</div>', unsafe_allow_html=True)
        primary_icons = {
            "Home": "⌂  Home",
            "Live Prediction": "⚡  Live Prediction",
            "Project Workflow": "▦  Project Workflow",
        }
        primary_page = st.radio(
            "Navigate",
            PRIMARY_PAGES,
            key="primary_navigation",
            format_func=lambda option: primary_icons[option],
            label_visibility="collapsed",
        )

        page = primary_page
        if primary_page == "Project Workflow":
            st.markdown(
                '<div class="workflow-nav">PROJECT SECTIONS</div>',
                unsafe_allow_html=True,
            )
            workflow_page = st.radio(
                "Project sections",
                WORKFLOW_PAGES,
                key="workflow_navigation",
                label_visibility="collapsed",
            )
            page = "Project Workflow" if workflow_page == "Workflow Overview" else workflow_page

        st.divider()
        st.markdown(
            '<div class="model-badge"><strong>Deployment model</strong><br>'
            'final_resnet50.keras<br>Decision threshold · 0.50</div>',
            unsafe_allow_html=True,
        )
        st.caption("Academic prototype")
    return page


def page_header(kicker: str, title: str, copy: str) -> None:
    """Render consistent page heading markup."""
    st.markdown(
        f'<div class="page-kicker">{kicker}</div><div class="page-title">{title}</div>'
        f'<div class="page-copy">{copy}</div>',
        unsafe_allow_html=True,
    )


def panel(title: str, copy: str) -> None:
    """Render a compact informational panel."""
    st.markdown(
        f'<div class="panel"><div class="panel-title">{title}</div>'
        f'<div class="panel-copy">{copy}</div></div>',
        unsafe_allow_html=True,
    )


def artifact_image(relative_path: str, caption: str) -> None:
    """Display an image artifact or a clear missing-artifact state."""
    path = PROJECT_ROOT / relative_path
    if path.is_file():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Artifact not found: {Path(relative_path).name}")


def metric_value(row: pd.Series, key: str) -> str:
    """Format a zero-to-one metric as a percentage."""
    return f"{float(row[key]):.1%}"


def final_metrics() -> pd.Series | None:
    """Return the tuned ResNet50 full-validation row."""
    data = load_csv_safely("outputs/model_comparison/tuned_model_metrics.csv")
    if data is None:
        return None
    rows = data[data["Model"] == "ResNet50"]
    return rows.iloc[0] if not rows.empty else None


def render_home_page() -> None:
    """Render project overview and verified headline results."""
    page_header(
        "Computer vision safety prototype",
        "SafeVison",
        "An end-to-end binary image-classification project covering dataset analysis, transfer learning, fine-tuning, evaluation, calibration, error analysis, explainability, and Streamlit deployment.",
    )
    st.markdown(
        "<div class='pipeline'>" + "".join(
            f"<span class='pipeline-step'>{step}</span>"
            + ("<span class='pipeline-arrow'>→</span>" if i < 8 else "")
            for i, step in enumerate(["Dataset", "EDA", "Preprocessing", "Training", "Fine-tuning", "Evaluation", "Calibration", "Error analysis", "Grad-CAM"])
        ) + "</div>",
        unsafe_allow_html=True,
    )
    row = final_metrics()
    if row is not None:
        cols = st.columns(5)
        for col, label, key in zip(cols, ["Accuracy", "Fall precision", "Fall recall", "Fall F1", "ROC-AUC"], ["Accuracy", "Precision_Fall", "Recall_Fall", "F1_Score_Fall", "ROC_AUC"]):
            col.metric(label, metric_value(row, key))
        st.caption("Tuned ResNet50 | full 361-image validation set | threshold 0.50")
    else:
        st.info("Final metrics artifact is unavailable.")
    st.markdown("### Project focus")
    left, middle, right = st.columns(3)
    with left:
        panel("Problem", "Recognize Fall versus No Fall from a single RGB image to support visual safety monitoring research.")
    with middle:
        panel("Selected model", "Fine-tuned ResNet50 with ImageNet initialization, class weighting, and a sigmoid P(Fall) output.")
    with right:
        panel("Deployment", "Cached Keras inference from streamlit_app/model/final_resnet50.keras at 224 x 224 x 3.")
    st.warning("Academic prototype only. It is not a certified medical device or emergency-response system.")


def clear_prediction() -> None:
    """Reset stored prediction state and force fresh uploader widgets."""
    st.session_state.prediction = None
    st.session_state.upload_version = st.session_state.get("upload_version", 0) + 1


def render_prediction_page() -> None:
    """Render controlled image inference with diagnostics and clear outcomes."""
    page_header("Deployment", "Live Prediction", "Test the packaged ResNet50 model with an uploaded image or a camera frame. Inference runs only when Analyse Image is selected.")
    st.info("The classifier has no person-detection stage. An image without a visible person can still receive a prediction.", icon=":material/person_alert:")
    if "upload_version" not in st.session_state:
        st.session_state.upload_version = 0
    source = st.segmented_control("Image source", ["Upload", "Camera"], default="Upload")
    if source == "Camera":
        image_file = st.camera_input("Capture an image", key=f"camera_{st.session_state.upload_version}")
    else:
        image_file = st.file_uploader("Upload JPG, JPEG, PNG, or WebP", type=["jpg", "jpeg", "png", "webp"], key=f"upload_{st.session_state.upload_version}")
    action, reset, _ = st.columns([1, 1, 4])
    analyse = action.button("Analyse Image", type="primary", icon=":material/search:", use_container_width=True)
    reset.button("Clear", icon=":material/refresh:", on_click=clear_prediction, use_container_width=True)
    if analyse:
        if image_file is None:
            st.warning("Select an image before running analysis.")
        else:
            try:
                raw = image_file.getvalue()
                opened = Image.open(io.BytesIO(raw))
                original_size = opened.size
                preview, batch = preprocess_image(opened)
                started = time.perf_counter()
                probability = predict_image(load_model(), batch)
                elapsed_ms = (time.perf_counter() - started) * 1000
                st.session_state.prediction = {"image": preview, "fall": probability, "elapsed": elapsed_ms, "size": original_size}
            except (UnidentifiedImageError, OSError):
                st.error("The selected file is not a readable image.")
            except Exception as exc:
                st.error("The model could not process this image.")
                with st.expander("Developer details"):
                    st.code(str(exc))
    result = st.session_state.get("prediction")
    if not result:
        st.markdown("<div class='status-line'>Ready for an image. The model will remain cached after its first load.</div>", unsafe_allow_html=True)
        return
    image_col, result_col = st.columns([1.3, 1])
    with image_col:
        st.image(result["image"], caption="Input image", use_container_width=True)
    fall_p = result["fall"]
    no_fall_p = 1.0 - fall_p
    is_fall = fall_p >= DECISION_THRESHOLD
    confidence = max(fall_p, no_fall_p)
    with result_col:
        st.markdown("### Detection result")
        if is_fall:
            st.error("Fall detected", icon=":material/warning:")
        else:
            st.success("No Fall detected", icon=":material/check_circle:")
        a, b = st.columns(2)
        a.metric("Fall probability", f"{fall_p:.1%}")
        b.metric("No Fall probability", f"{no_fall_p:.1%}")
        st.progress(fall_p, text=f"Fall {fall_p:.1%}")
        st.progress(no_fall_p, text=f"No Fall {no_fall_p:.1%}")
        st.metric("Prediction confidence", f"{confidence:.1%}")
    with st.expander("Technical prediction details"):
        details = pd.DataFrame({"Property": ["Decision threshold", "Processing time", "Original dimensions", "Model input", "Output meaning", "Preprocessing"], "Value": ["0.50", f"{result['elapsed']:.1f} ms", f"{result['size'][0]} x {result['size'][1]}", "224 x 224 x 3 RGB", "Sigmoid = P(Fall)", "Embedded ResNet50 preprocessing"]})
        st.dataframe(details, hide_index=True, use_container_width=True)


def render_workflow_page() -> None:
    """Map verified notebook responsibilities across the ML lifecycle."""
    page_header("Method", "Project Workflow", "The repository records the complete progression from exploratory analysis to a deployable, explainable classifier.")
    stages = [
        ("1. Dataset audit", "Raw image folders", "Class counts, dimensions, quality and duplicates", "EDA findings", "01_EDA.ipynb; blur_leakage_test.ipynb"),
        ("2. Baseline development", "224 x 224 RGB images", "Custom CNN and transfer-learning baselines", "Saved baseline models", "custom_cnn_training.ipynb; model_training.ipynb; resnet50_training.ipynb"),
        ("3. Baseline comparison", "Three baseline models", "Common validation evaluation", "Metrics, timing and confusion matrices", "05_Baseline_Model_Comparison.ipynb"),
        ("4. Hyperparameter training", "Train/validation datasets", "Augmentation, class weights, frozen heads and fine-tuning", "Tuned models and CSV histories", "06_Hyperparameter_Training.ipynb"),
        ("5. Error and calibration analysis", "Final probabilities", "Error slicing and temperature search", "Calibration and error artifacts", "03_Error_Analysis.ipynb"),
        ("6. Explainability", "Correct and incorrect predictions", "Gradient-weighted activation maps", "Eight Grad-CAM panels", "04_GradCAM_Analysis.ipynb"),
        ("7. Deployment", "Final ResNet50 artifact", "Cached inference and dashboard presentation", "Streamlit application", "streamlit_app/app.py"),
    ]
    for title, input_, operation, output, notebook in stages:
        with st.expander(title, expanded=title.startswith("1")):
            cols = st.columns(4)
            cols[0].markdown(f"**Input**\n\n{input_}")
            cols[1].markdown(f"**Operation**\n\n{operation}")
            cols[2].markdown(f"**Output**\n\n{output}")
            cols[3].markdown(f"**Evidence**\n\n`{notebook}`")


def render_dataset_page() -> None:
    """Render dataset composition, mapping, representative samples, and EDA notes."""
    page_header("Data", "Dataset & EDA", "The processed dataset combines fall and activity imagery into an explicit project reporting convention: 0 = No Fall and 1 = Fall.")
    counts = pd.DataFrame({"Split": ["Train", "Train", "Validation", "Validation"], "Class": ["No Fall", "Fall", "No Fall", "Fall"], "Images": [684, 439, 215, 146]})
    cols = st.columns(4)
    cols[0].metric("Training images", "1,123")
    cols[1].metric("Validation images", "361")
    cols[2].metric("No Fall mapping", "0")
    cols[3].metric("Fall mapping", "1")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### Class distribution")
        chart = counts.pivot(index="Split", columns="Class", values="Images")
        st.bar_chart(chart, color=["#35a7d7", "#ef6b73"], height=320)
        st.caption("Counts verified from data/processed. The training set is moderately imbalanced, motivating balanced class weights.")
    with right:
        st.markdown("### EDA findings")
        st.dataframe(counts, hide_index=True, use_container_width=True)
        st.markdown("- Images are standardized to 224 x 224 at dataset loading time.\n- Horizontal flips, small rotations, zoom, and contrast shifts are applied during tuned training.\n- `01_EDA.ipynb` records image inspection and source consolidation.\n- `blur_leakage_test.ipynb` documents the dedicated blur/leakage check.")
    st.markdown("### Validation examples")
    examples = [APP_DIR / "assets/no_fall_1.png", APP_DIR / "assets/no_fall_2.png", APP_DIR / "assets/fall_1.png", APP_DIR / "assets/fall_2.jpg"]
    labels = ["No Fall", "No Fall", "Fall", "Fall"]
    sample_cols = st.columns(4)
    for col, path, label in zip(sample_cols, examples, labels):
        image = load_image_safely(path)
        if image is not None:
            col.image(image, caption=label, use_container_width=True)
        else:
            col.info(f"{label} sample unavailable")


def render_models_page() -> None:
    """Compare the verified architecture strategies used by the project."""
    page_header("Design", "Model Architectures", "Three model families test the trade-off between a learned-from-scratch baseline, efficient transfer learning, and deeper residual features.")
    tabs = st.tabs(["Custom CNN", "MobileNetV3", "ResNet50"])
    specs = [
        ("Custom CNN", "From scratch", "Conv blocks: 32, 64, 128, 256; batch normalization; max pooling", "Global average pooling -> Dense 128 -> Dropout 0.40 -> sigmoid", "Small artifact and complete architectural control", "Weakest tuned validation performance; limited representation capacity"),
        ("MobileNetV3 Small", "ImageNet transfer learning", "include_top=False; built-in preprocessing; frozen backbone then last 30 layers", "Global average pooling -> Dense 128 -> Dropout 0.30 -> sigmoid", "Compact, fast, strong fall recall", "Lower precision and accuracy than ResNet50"),
        ("ResNet50", "ImageNet transfer learning", "include_top=False; embedded ResNet50 preprocessing; frozen backbone then last 30 layers", "Global average pooling -> Dense 256 -> Dropout 0.40 -> sigmoid", "Residual skip connections support deeper features; best tuned precision and accuracy", "Largest model and slower inference; tuned recall fell versus baseline"),
    ]
    for tab, spec in zip(tabs, specs):
        with tab:
            name, transfer, backbone, head, advantage, limitation = spec
            a, b, c = st.columns(3)
            a.metric("Input", "224 x 224 x 3")
            b.metric("Output", "1 sigmoid")
            b.caption("P(Fall)")
            sizes = {"Custom CNN": "4.93 MB baseline", "MobileNetV3 Small": "4.15 MB baseline", "ResNet50": "90.65 MB deployment"}
            c.metric("Artifact size", sizes[name])
            st.markdown(f"**Training status:** {transfer}\n\n**Feature path:** {backbone}\n\n**Classification head:** {head}")
            left, right = st.columns(2)
            with left:
                panel("Advantage", advantage)
            with right:
                panel("Limitation", limitation)
            if name == "ResNet50":
                with st.expander("Inspect deployment model"):
                    if st.button("Load architecture details", icon=":material/memory:"):
                        model = load_model()
                        st.metric("Parameters", f"{model.count_params():,}")
                        stream = io.StringIO()
                        model.summary(print_fn=lambda line: stream.write(line + "\n"))
                        st.code(stream.getvalue(), language=None)


def history_chart(paths: list[str], metrics: list[str], caption: str) -> None:
    """Combine available history series and render a Streamlit line chart."""
    frames: list[pd.DataFrame] = []
    offset = 0
    for path in paths:
        frame = load_history(path)
        if frame is None:
            continue
        usable = [column for column in metrics if column in frame.columns]
        if not usable:
            continue
        part = frame[usable].copy()
        part.index = np.arange(offset + 1, offset + len(part) + 1)
        offset += len(part)
        frames.append(part)
    if frames:
        st.line_chart(pd.concat(frames), height=310)
        st.caption(caption)
    else:
        st.info("Saved training history is unavailable for this chart.")


def render_training_page() -> None:
    """Render verified tuned-training configuration and saved histories."""
    page_header("Optimization", "Training & Hyperparameters", "Configurations below are read from the tuned-training notebook and saved CSV histories; no model is retrained by this app.")
    config = pd.DataFrame({
        "Model": ["Custom CNN", "MobileNetV3", "ResNet50"], "Batch": [32, 32, 16], "Head LR": ["1e-3", "1e-3", "1e-3"], "Fine-tune LR": ["n/a", "1e-5", "1e-5"], "Max epochs": [30, 30, 30], "Dropout": [0.40, 0.30, 0.40]
    })
    st.dataframe(config, hide_index=True, use_container_width=True)
    a, b, c = st.columns(3)
    with a:
        panel("Loss and optimizer", "Binary cross-entropy with Adam. Metrics include accuracy, precision, recall, ROC-AUC and PR-AUC.")
    with b:
        panel("Transfer phases", "Up to 15 frozen-backbone epochs, followed by up to 15 fine-tuning epochs for the last 30 backbone layers.")
    with c:
        panel("Class weights", "Computed with balanced weighting: approximately 0.821 for No Fall and 1.279 for Fall.")
    with st.expander("Callbacks and augmentation", expanded=True):
        st.markdown("**Callbacks:** best checkpoint on `val_pr_auc`; early stopping patience 5 and min delta 0.001; ReduceLROnPlateau on `val_loss`, factor 0.2, patience 2, floor 1e-7; CSVLogger.\n\n**Augmentation:** horizontal flip, rotation 0.05, zoom 0.10, and contrast 0.10.")
    model = st.selectbox("Training history", ["ResNet50", "MobileNetV3", "Custom CNN"])
    history_paths = {
        "ResNet50": ["histories/hyperparameter_training/tuned_resnet50_phase1.csv", "histories/hyperparameter_training/tuned_resnet50_phase2.csv"],
        "MobileNetV3": ["histories/hyperparameter_training/tuned_mobilenetv3_phase1.csv", "histories/hyperparameter_training/tuned_mobilenetv3_phase2.csv"],
        "Custom CNN": ["histories/hyperparameter_training/tuned_custom_cnn.csv"],
    }
    left, right = st.columns(2)
    with left:
        history_chart(history_paths[model], ["accuracy", "val_accuracy"], f"Saved {model} training and validation accuracy.")
    with right:
        history_chart(history_paths[model], ["loss", "val_loss"], f"Saved {model} training and validation loss.")


def comparison_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Select and format the common model comparison columns."""
    columns = ["Model", "Accuracy", "Precision_Fall", "Recall_Fall", "F1_Score_Fall", "ROC_AUC", "PR_AUC"]
    view = frame[columns].copy()
    for column in columns[1:]:
        view[column] = view[column].map(lambda value: f"{value:.1%}")
    return view.rename(columns={"Precision_Fall": "Fall precision", "Recall_Fall": "Fall recall", "F1_Score_Fall": "Fall F1", "ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC"})


def render_comparison_page() -> None:
    """Render separate baseline and tuned full-validation comparisons."""
    page_header("Selection", "Model Comparison", "All rows on this page use the same 361-image validation set and a 0.50 decision threshold.")
    baseline = load_csv_safely("outputs/model_comparison/baseline_model_metrics.csv")
    tuned = load_csv_safely("outputs/model_comparison/tuned_model_metrics.csv")
    baseline_tab, tuned_tab = st.tabs(["Baseline models", "Tuned models"])
    with baseline_tab:
        if baseline is not None:
            st.dataframe(comparison_table(baseline), hide_index=True, use_container_width=True)
            artifact_image("outputs/model_comparison/baseline_model_comparison.png", "Baseline validation comparison generated by the project.")
        else:
            st.info("Baseline metrics artifact not found.")
    with tuned_tab:
        if tuned is not None:
            st.dataframe(comparison_table(tuned), hide_index=True, use_container_width=True)
            chart = tuned.set_index("Model")[["Accuracy", "Precision_Fall", "Recall_Fall", "F1_Score_Fall"]]
            st.bar_chart(chart, color=["#35a7d7", "#46b889", "#e0a43a", "#ef6b73"], height=360)
        else:
            st.info("Tuned metrics artifact not found.")
    st.markdown("### Selection rationale")
    st.markdown("ResNet50 was selected for its 92.8% tuned accuracy, 97.6% fall precision, and 99.0% ROC-AUC. MobileNetV3 remains the lightweight alternative with stronger fall recall than tuned ResNet50. The Custom CNN improved substantially after tuning but remained capacity-limited. Fine-tuning shifted ResNet50 toward far fewer false positives (39 to 3) while increasing false negatives (4 to 23), raising precision and reducing recall.")


def render_evaluation_page() -> None:
    """Render final full-validation results and class-error interpretation."""
    page_header("Results", "Final Model Evaluation", "The selected tuned ResNet50 is evaluated below on the complete 361-image validation set. Calibration results are kept on their own page because they use a separate subset.")
    row = final_metrics()
    if row is None:
        st.info("Final evaluation artifact not found.")
        return
    cols = st.columns(6)
    for col, label, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"], ["Accuracy", "Precision_Fall", "Recall_Fall", "F1_Score_Fall", "ROC_AUC", "PR_AUC"]):
        col.metric(label, metric_value(row, key))
    left, right = st.columns([1, 1.15])
    with left:
        st.markdown("### Confusion matrix")
        matrix = pd.DataFrame([[int(row["True_Negative"]), int(row["False_Positive"])], [int(row["False_Negative"]), int(row["True_Positive"])]], index=["Actual No Fall", "Actual Fall"], columns=["Predicted No Fall", "Predicted Fall"])
        st.dataframe(matrix, use_container_width=True)
        st.caption("Full validation set: TN 212, FP 3, FN 23, TP 123. Threshold 0.50.")
    with right:
        st.markdown("### Reading the metrics")
        st.markdown("**Accuracy** measures all correct decisions. **Precision** asks how often a fall alert was correct. **Recall** asks how many actual falls were found. **F1** balances precision and recall. **ROC-AUC** measures ranking across thresholds; **PR-AUC** focuses on positive-class performance under imbalance.")
        st.warning("The 23 false negatives matter most operationally because they represent missed falls. The 3 false positives represent unnecessary alerts.")
    artifact_image("results/resnet50_confusion_matrix.png", "Baseline ResNet50 confusion matrix (88.09% evaluation), retained as a separate historical artifact.")
    st.info("No standalone tuned ROC or Precision-Recall curve image is present. Their verified area metrics are shown above from tuned_model_metrics.csv.")


def render_calibration_page() -> None:
    """Render temperature search and before/after confidence evaluation."""
    page_header("Confidence quality", "Calibration", "Temperature scaling was evaluated on a separate 181-image analysis subset. It adjusts confidence values without retraining the classifier.")
    summary = load_csv_safely("outputs/error_analysis/temperature_summary.csv")
    comparison = load_csv_safely("outputs/error_analysis/temperature_rmse_comparison.csv")
    if summary is not None:
        cols = st.columns(3)
        cols[0].metric("Selected temperature", f"{summary.iloc[0]['Best_Temperature']:.2f}")
        cols[1].metric("Calibration log-loss", f"{summary.iloc[0]['Calibration_Log_Loss']:.4f}")
        cols[2].metric("Threshold", f"{summary.iloc[0]['Classification_Threshold']:.2f}")
    if comparison is not None:
        st.dataframe(comparison.round(4), hide_index=True, use_container_width=True)
        st.caption("On this subset, class metrics remained unchanged. RMSE moved from 0.2275 to 0.2295 and log-loss from 0.1737 to 0.1768; the searched scaling did not improve these reported error measures.")
    tab1, tab2, tab3 = st.tabs(["Temperature search", "RMSE and log-loss", "Selection"])
    with tab1:
        artifact_image("outputs/error_analysis/temperature_search_curve.png", "Log-loss across the searched temperature range.")
    with tab2:
        artifact_image("outputs/error_analysis/rmse_logloss_comparison.png", "Before/after RMSE and log-loss comparison.")
    with tab3:
        artifact_image("outputs/error_analysis/temperature_selection.png", "Selected temperature from the project search.")
    st.markdown("Raw probabilities are clipped, converted to logits, divided by a candidate temperature, and transformed back through the sigmoid. In the usual binary formulation this changes confidence but preserves ranking, so ROC-AUC and the ordering of examples remain unchanged.")


def render_error_analysis_page() -> None:
    """Render the calibrated subset's error inventory and safety implications."""
    page_header("Failure modes", "Error Analysis", "Error artifacts below come from the 181-image calibrated analysis subset, not the complete 361-image tuned validation report.")
    analysis = load_csv_safely("outputs/error_analysis/prediction_error_analysis.csv")
    fp = load_csv_safely("outputs/error_analysis/false_positives.csv")
    fn = load_csv_safely("outputs/error_analysis/false_negatives.csv")
    if analysis is not None and "Prediction_Type" in analysis:
        counts = analysis["Prediction_Type"].value_counts()
        cols = st.columns(4)
        for col, name in zip(cols, ["True Positive", "True Negative", "False Positive", "False Negative"]):
            col.metric(name, int(counts.get(name, 0)))
        st.bar_chart(counts, color="#35a7d7", horizontal=True, height=280)
    artifact_image("outputs/error_analysis/calibrated_confusion_matrix.png", "Confusion matrix for the separate calibrated analysis subset.")
    fp_tab, fn_tab = st.tabs(["False positives", "False negatives"])
    with fp_tab:
        st.markdown("A false positive reports a fall when the person is not falling. It can create unnecessary alerts.")
        if fp is not None:
            st.dataframe(fp, hide_index=True, use_container_width=True)
        else:
            st.info("False-positive table not found.")
    with fn_tab:
        st.markdown("A false negative misses an actual fall. In a safety-oriented system this is the more serious failure mode.")
        if fn is not None:
            st.dataframe(fn, hide_index=True, use_container_width=True)
        else:
            st.info("False-negative table not found.")


def render_gradcam_page() -> None:
    """Render saved Grad-CAM evidence without expensive recomputation."""
    page_header("Explainability", "Grad-CAM Explainability", "Saved activation overlays show which spatial regions influenced correct and incorrect ResNet50 decisions.")
    results = load_csv_safely("outputs/gradcam/final_gradcam_results.csv")
    if results is None:
        st.info("Grad-CAM result table not found.")
        return
    categories = list(results["category"].dropna().unique())
    selected = st.selectbox("Prediction category", categories)
    subset = results[results["category"] == selected].reset_index(drop=True)
    cols = st.columns(2)
    for index, row in subset.iterrows():
        filename = Path(str(row["save_path"]).replace("\\", "/")).name
        path = PROJECT_ROOT / "outputs/gradcam" / filename
        with cols[index % 2]:
            if path.is_file():
                st.image(str(path), use_container_width=True)
            st.markdown(f"**{row['actual_class']} -> {row['predicted_class']}** | confidence {float(row['confidence']):.1%}")
            st.caption(str(row["observation"]))
    with st.expander("How Grad-CAM works"):
        st.markdown("The project targets the final convolutional representation, computes gradients of the selected output with respect to feature maps, globally averages those gradients into channel weights, and combines the weighted maps. ReLU keeps positive evidence; the heatmap is resized and overlaid on the source image. It is useful for inspecting attention, but it is a coarse explanation and does not prove causal reasoning.")


def render_blockers_page() -> None:
    """Render the documented engineering investigation history."""
    page_header("Engineering log", "Blockers & Solutions", "The most valuable work was often in validating assumptions around labels, probabilities, evaluation scope, and runtime behavior.")
    blockers = [
        ("Directory class order", "TensorFlow discovered ['fall', 'no_fall'], which conflicted with reporting.", "Alphabetical directory indexing.", "Explicitly supplied class_names=['no_fall', 'fall'] so 0 = No Fall and 1 = Fall.", "Never infer business semantics from folder order."),
        ("Probability direction", "Early interpretation inverted predictions and produced ROC-AUC near 0.0098.", "The sigmoid direction was assumed instead of traced through remapped labels.", "Evaluation verified the output as P(Fall); corrected ROC-AUC rose to the expected direction.", "Validate probability semantics with known examples and AUC."),
        ("Missing history object", "history.history raised NameError in an evaluation notebook.", "The notebook loaded a model but never called model.fit().", "Persisted CSV histories and plot artifacts are loaded instead.", "Training state must be serialized for later analysis."),
        ("Evaluation scope mismatch", "A [[107, 1], [10, 63]] matrix did not match full-validation totals.", "Calibration and tuned evaluation used different subsets.", "Every dashboard view now labels its dataset scope.", "Never compare metrics without confirming denominators."),
        ("Native Windows GPU", "Recent TensorFlow builds ran on CPU despite an NVIDIA GPU.", "TensorFlow after 2.10 does not provide native-Windows CUDA support.", "Use CPU locally or WSL2 for supported GPU acceleration.", "Platform support is part of ML system design."),
    ]
    for problem, symptom, cause, fix, lesson in blockers:
        with st.expander(problem):
            st.markdown(f"**Symptom**\n\n{symptom}\n\n**Root cause / investigation**\n\n{cause}\n\n**Fix**\n\n{fix}\n\n**Lesson**\n\n{lesson}")


def render_stack_page() -> None:
    """Render verified tools grouped by their role in the project."""
    page_header("Implementation", "Technology Stack", "Libraries are grouped by the work they perform in the repository, rather than mixing frameworks with model concepts.")
    groups = [
        ("Core", "Python; Jupyter Notebook", "Experiment orchestration, repeatable analysis, and saved notebook evidence."),
        ("Deep learning", "TensorFlow; Keras", "Image datasets, Keras models, ResNet50, MobileNetV3, custom layers, callbacks, and inference."),
        ("Data", "NumPy; Pandas", "Tensor preparation, probability manipulation, structured metrics, histories, and error tables."),
        ("Evaluation", "Scikit-learn", "Class weights, confusion matrices, classification reports, ROC/PR metrics, and loss analysis."),
        ("Visualization", "Matplotlib; Seaborn", "Training curves, model comparisons, matrices, calibration, and explainability figures."),
        ("Images", "Pillow", "RGB conversion, EXIF orientation, resizing, image preview, and saved examples."),
        ("Deployment", "Streamlit", "Cached model serving, navigation, interactive charts, upload/camera input, and responsive UI."),
    ]
    for start in range(0, len(groups), 3):
        cols = st.columns(3)
        for col, (group, tools_, use) in zip(cols, groups[start:start + 3]):
            with col:
                panel(group, f"<strong>{tools_}</strong><br><br>{use}")
        st.write("")


def render_future_page() -> None:
    """Separate current system constraints from proposed future work."""
    page_header("Boundaries", "Limitations & Future Scope", "This page distinguishes what the current image classifier actually does from credible extensions that have not yet been implemented.")
    limits, future = st.columns(2)
    with limits:
        st.markdown("### Current limitations")
        for item in ["Single-image classification; no temporal motion understanding", "No person-detection stage", "No automatic emergency-alert integration", "Dataset and camera-angle bias", "Sensitivity to occlusion and unusual posture", "Predictions still occur when no person is visible", "Academic prototype without clinical or safety certification", "ResNet50 compute and latency cost"]:
            st.markdown(f"- {item}")
    with future:
        st.markdown("### Future scope")
        for item in ["Real-time video and webcam inference", "Person detection before classification", "CNN-LSTM, temporal transformer, or 3D CNN evaluation", "Larger, more diverse datasets", "Edge deployment, quantization, and pruning", "Configurable alerts and escalation workflows", "Model drift and confidence monitoring", "Safety-weighted threshold tuning and stronger calibration"]:
            st.markdown(f"- {item}")


def render_interview_page() -> None:
    """Render concise project-specific interview revision answers."""
    page_header("Preparation", "Interview Quick Revision", "Project-specific answers for explaining design decisions, results, trade-offs, and next steps without drifting into generic definitions.")
    answers = {
        "Project in 30 seconds": "I built SafeVison, a binary human fall detection image classifier, and compared a Custom CNN, MobileNetV3, and ResNet50. After transfer learning and fine-tuning, ResNet50 reached 92.8% accuracy, 97.6% fall precision, and 99.0% ROC-AUC on 361 validation images. I then analyzed errors, tested temperature scaling, used Grad-CAM, and deployed the final model with Streamlit.",
        "Project in 2 minutes": "The workflow begins with dataset consolidation, EDA, duplicate and blur checks, and an explicit 0 = No Fall / 1 = Fall mapping. I trained from-scratch and transfer-learning baselines, then tuned all three models with balanced class weights and augmentation. MobileNetV3 was efficient and recall-oriented; ResNet50 was selected for its overall tuned accuracy and precision. I kept calibration analysis separate because it used a different subset, inspected false alerts and missed falls, and used Grad-CAM to see whether attention aligned with the person and floor-contact regions. The deployed model embeds ResNet preprocessing and returns P(Fall).",
        "Why ResNet50?": "It provided the strongest tuned validation result and excellent probability ranking. Residual connections support deep feature learning while ImageNet weights reduce the amount of task-specific data required.",
        "Why transfer learning?": "The processed training set has 1,123 images, so learning all visual features from scratch is difficult. ImageNet initialization supplies reusable edges, textures, shapes, and object-level representations.",
        "Why freeze, then fine-tune?": "The frozen phase trains a stable classification head without disrupting pretrained features. Later, only the last 30 backbone layers are unfrozen at 1e-5 so higher-level features can adapt gradually.",
        "Why class weights?": "Training contains 684 No Fall and 439 Fall images. Balanced weights increase the loss contribution of the less frequent Fall class.",
        "Precision versus recall": "The tuned ResNet50 has 97.6% fall precision but 84.2% recall. Alerts are usually correct, but 23 of 146 validation falls were missed. A safety deployment may choose a lower threshold to favor recall.",
        "Why false negatives matter": "A false negative is a missed actual fall, so no assistance may be triggered. That consequence is more serious than the inconvenience of many false alerts.",
        "Why ROC-AUC and PR-AUC?": "ROC-AUC checks ranking across thresholds, while PR-AUC focuses on positive-class precision and recall under imbalance. ResNet50 achieved about 99.0% ROC-AUC and 98.5% PR-AUC after tuning.",
        "What is temperature scaling?": "It divides logits by a learned temperature before applying sigmoid. It changes confidence without changing ranking. In this project, the searched result did not improve the reported RMSE or log-loss, which is important to report honestly.",
        "What is Grad-CAM?": "It uses output gradients to weight convolutional feature maps, producing a coarse heatmap of positive evidence. The saved examples reveal attention on posture, limbs, floor contact, and sometimes misleading background regions.",
        "Main blocker and solution": "The key blocker was probability direction after class remapping. An implausible ROC-AUC near 0.0098 exposed the inversion. Tracing the explicit class order and testing known samples confirmed the sigmoid output is P(Fall).",
        "Main limitation": "It is a still-image classifier with no person detector or temporal model, so it cannot reason about motion and will predict even for images without a person.",
        "What comes next?": "Add person detection and temporal video modeling, optimize for edge inference, expand the dataset, and tune the operating threshold around the cost of missed falls.",
    }
    for question, answer in answers.items():
        with st.expander(question):
            st.write(answer)


PAGE_RENDERERS: dict[str, Any] = {
    "Home": render_home_page,
    "Live Prediction": render_prediction_page,
    "Project Workflow": render_workflow_page,
    "Dataset & EDA": render_dataset_page,
    "Model Architectures": render_models_page,
    "Training & Hyperparameters": render_training_page,
    "Model Comparison": render_comparison_page,
    "Final Model Evaluation": render_evaluation_page,
    "Calibration": render_calibration_page,
    "Error Analysis": render_error_analysis_page,
    "Grad-CAM Explainability": render_gradcam_page,
    "Blockers & Solutions": render_blockers_page,
    "Technology Stack": render_stack_page,
    "Limitations & Future Scope": render_future_page,
    "Interview Quick Revision": render_interview_page,
}


def main() -> None:
    """Run the dashboard."""
    page = render_sidebar()
    apply_theme(st.session_state.theme)
    PAGE_RENDERERS[page]()


if __name__ == "__main__":
    main()
