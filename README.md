# 🔥 FumeWatch: Real-Time Smoke & Fire Detection System

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![YOLO](https://img.shields.io/badge/YOLO-v8-FF6F00.svg)](https://ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg)](https://opencv.org/)
[![Debian Trixie](https://img.shields.io/badge/Streamlit_Cloud-Debian_Trixie-0055A5.svg)](https://share.streamlit.io/)

**FumeWatch** is an AI-powered computer vision application designed for early detection of smoke and fire hazards. Built using **YOLOv8**, **OpenCV**, and **Streamlit**, it provides an intuitive web interface for uploading images and processing video streams to detect environmental threats instantly with web-compatible video encoding (H.264).

---

## 📌 Project Overview

* **Project Name:** FumeWatch
* **Task:** Object Detection (Smoke & Fire)
* **Dataset:** Kaggle [(`sayedgamal99/smoke-fire-detection-yolo`)](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo/data)
* **Classes:**
  * `0`: **Smoke** (Blue Box)
  * `1`: **Fire** (Red Box)
* **Deployment Target:** Streamlit Community Cloud

---

## 📌 Tech Stack

* **Core AI / ML:** Python, Ultralytics YOLOv8, OpenCV, NumPy
* **Data Processing & EDA:** Pandas, Matplotlib, Seaborn
* **Media Processing & Transcoding:** OpenCV, FFmpeg (`libx264` Web Video Encoding)
* **Frontend & Deployment:** Streamlit Community Cloud

---

## 📌 Repository Structure

```text
FumeWatch/
├── notebook/                  # Notebooks for EDA and Model Training with showing results
│   └── simple-fire-detection.ipynb 
│
├── weights/                   # Trained Model Weights
│   └── best.pt                # YOLO custom trained weights
│
├── utils.py                   # Image preprocessing, box plotting, and video transcoding
├── model.py                   # Model loading and inference wrapper
├── app.py                     # Streamlit web application dashboard
├── packages.txt               # System-level dependencies for Streamlit Cloud (ffmpeg, libgl1)
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md