# 🔥 FumeWatch: Real-Time Smoke & Fire Detection System

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.0-61DAFB.svg)](https://reactjs.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v8-FF6F00.svg)](https://ultralytics.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)

**FumeWatch** is an AI-powered computer vision application designed for early detection of smoke and fire hazards. Built using **YOLO**, **FastAPI**, and **Streamlit**, it provides an intuitive web interface for uploading images or real-time media streams to detect environmental threats instantly.

---

## 📌 Project Overview

* **Project Name:** FumeWatch
* **Task:** Object Detection (Smoke & Fire)
* **Dataset:** Kaggle [(`sayedgamal99/smoke-fire-detection-yolo`)](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo/data)
* **Classes:**
  * `0`: **Smoke** (Blue Box)
  * `1`: **Fire** (Red Box)
* **Deployment Target:** Streamlit Community Cloud / Docker

---

## 📌 Tech Stack

* **Core AI / ML:** Python, Ultralytics YOLO, OpenCV, NumPy
* **Data Processing & EDA:** Pandas, Matplotlib, Seaborn
* **Backend API:** FastAPI, Uvicorn (Optional for modular deployment)
* **Frontend / Web UI:** Streamlit

---

## 📌 Repository Structure

```text
FumeWatch/
├── notebook/                  # Notebooks for EDA and Model Training with showing results
│   └── simple-fire-detection.ipynb 
│
├── weights/                  # Trained Model Weights
│   └── best.pt               # YOLO custom trained weights
│
├── utils.py                  # Utility functions has functions for processing
│
├── model.py                  # Model loading and inference functions
│
├── app.py                    # Main application file
│
├── .gitignore
└── README.md```