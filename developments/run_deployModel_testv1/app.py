import streamlit as st
from PIL import Image
import pandas as pd
from collections import Counter

from predictor import load_model, run_inference, draw_results, parse_detections

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traditional Building Detector",
    page_icon="🏛️",
    layout="wide"
)

st.title("🏛️ Traditional Building Component Detector")
st.caption("Detects and segments columns, doors, windows, and other structural elements")

# ── Sidebar Controls ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence to show a detection"
    )

    iou_threshold = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Non-max suppression threshold"
    )

    show_masks = st.toggle("Show Segmentation Masks", value=True)
    show_table = st.toggle("Show Detection Table", value=True)

    st.divider()
    st.markdown("**Model:** YOLOv11 (Roboflow)")
    st.markdown("**Task:** Instance Segmentation")

# ── Load Model (cached) ───────────────────────────────────────────────────
@st.cache_resource
def get_model():
    with st.spinner("Loading YOLOv11 model..."):
        return load_model("saved_model/model.pt")

model = get_model()
st.sidebar.success("✅ Model ready")

# ── Upload Image ──────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a building image",
    type=["jpg", "jpeg", "png"],
    help="Upload a photo of a traditional building"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_column_width=True)

    # ── Run Inference ─────────────────────────────────────────────────────
    with st.spinner("🔍 Detecting building components..."):
        result = run_inference(model, image, conf_threshold=conf_threshold)
        annotated_image = draw_results(result)
        detections = parse_detections(result)

    with col2:
        st.subheader("🔍 Detection Result")
        st.image(annotated_image, use_column_width=True)

    # ── Summary Metrics ───────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Detection Summary")

    if len(detections) == 0:
        st.warning("No components detected. Try lowering the confidence threshold.")
    else:
        # Count per class
        class_counts = Counter(d["class_name"] for d in detections)

        # Show metric cards per class
        cols = st.columns(len(class_counts))
        for idx, (class_name, count) in enumerate(class_counts.items()):
            cols[idx].metric(
                label=f"🧱 {class_name.capitalize()}",
                value=count
            )

        # ── Detection Table ───────────────────────────────────────────────
        if show_table:
            st.subheader("📋 All Detections")
            df = pd.DataFrame([
                {
                    "Component":   d["class_name"].capitalize(),
                    "Confidence":  f"{d['confidence']*100:.1f}%",
                    "Has Mask":    "✅" if d["has_mask"] else "❌",
                    "BBox [x1,y1,x2,y2]": [round(v) for v in d["bbox_xyxy"]],
                }
                for d in detections
            ])
            st.dataframe(df, use_container_width=True)

        # ── Per-class confidence chart ─────────────────────────────────────
        st.subheader("📈 Confidence per Detection")
        chart_data = pd.DataFrame({
            "Component": [d["class_name"] for d in detections],
            "Confidence": [d["confidence"] for d in detections]
        })
        st.bar_chart(chart_data.set_index("Component"))