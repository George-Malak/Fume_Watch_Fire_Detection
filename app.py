import os
import tempfile
import cv2
import numpy as np
from PIL import Image
import streamlit as st
# import model and utils
from model import load_yolo_model
from utils import FumeWatchProcessor

# 1. set page config 
st.set_page_config(
    page_title="FumeWatch - Smoke & Fire Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("FumeWatch System")
st.write("AI-powered system for early detection of smoke and fire hazards through images and video clips.")

# 2. load model and processor
model = load_yolo_model()
processor = FumeWatchProcessor(target_size=(640, 640))

# 3. sidebar settings 
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.25, 
    step=0.05
)

app_mode = st.sidebar.radio("Choose Operation Mode:", ["Test Image", "Test Video"])

# 4. Test Image Mode
if app_mode == "Test Image":
    st.subheader("Detection on Images")
    uploaded_image = st.file_uploader("Upload an image for testing...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None and model is not None:
        # convert uploaded image to numpy array
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image_np = cv2.imdecode(file_bytes, 1)

        col1, col2 = st.columns(2)

        with col1:
            st.info("Original Uploaded Image")
            st.image(image_np, channels="BGR", use_container_width=True)

        with col2:
            st.info("Detection Results")
            with st.spinner("Processing image and making predictions..."):
                # call for prediction and get annotated image
                annotated_img, _ = processor.predict_image(image_np, model, conf_threshold=conf_threshold)
                st.image(annotated_img, use_container_width=True)

# 5. Test Video Mode 
elif app_mode == "Test Video":
    st.subheader("🎥 Detection on Video Clips")
    uploaded_video = st.file_uploader("Upload a video for testing...", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None and model is not None:
        # Save the uploaded video in a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()

        st.info("Video uploaded successfully. Click the button below to start processing.")
        
        output_video_path = "output_processed_video.mp4"

        if st.button("Start Video annotation"):
            with st.spinner("Processing video frames... This may take a few seconds/minutes depending on the video size."):
                success = processor.process_video(
                    video_path=tfile.name,
                    output_path=output_video_path,
                    model=model,
                    conf_threshold=conf_threshold
                )

            if success and os.path.exists(output_video_path):
                st.success("✓ Video processing completed successfully!")
                st.video(output_video_path)
            else:
                st.error("An error occurred while processing the video.")

        # Clear the temporary file after completion
        try:
            os.remove(tfile.name)
        except Exception:
            pass