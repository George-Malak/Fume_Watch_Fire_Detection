import streamlit as st
from ultralytics import YOLO

PTH_TO_MODEL  = "weights/best.pt"

@st.cache_resource
def load_yolo_model(model_path: str = PTH_TO_MODEL):
    """
    load YOLO model from cache and start use it.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"check again the best.pt file: {e}")
        return None