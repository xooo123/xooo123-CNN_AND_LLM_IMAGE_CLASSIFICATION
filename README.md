#  CNN + LLM COVID-19 X-Ray Classifier
A hybrid **Convolutional Neural Network** + **Multimodal Large Language Model (MLLM)** system for classifying and explaining **COVID-19 chest X-ray images**.

This project trains a custom CNN from scratch, evaluates it, serves it using **FastAPI**, and optionally generates natural-language explanations using an LLM.

---

##  Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [LLM Explanations](#llm-explanations)
- [FastAPI Server](#fastapi-server)
- [API Examples](#api-examples)
- [License](#license)

---

## 🧠 Overview

This project implements:

### **Custom CNN Model (from scratch)**
Defined in `src/model.py`, built with:
- 4 convolution blocks (32 → 64 → 128 → 256)
- BatchNorm + ReLU + MaxPool
- AdaptiveAvgPool
- Fully connected classifier with dropout

### **LLM Explanation System**
`src/llm_wrapper.py` generates medical-style explanations based on:
- Predicted class  
- Class probabilities  
- (optional) image source  

### **FastAPI Model Server**
- `/predict` — CNN inference + optional LLM explanation  
- `/health` — metadata  
- Serves frontend under `/static`

---

## 📂 Dataset

This project uses the **COVID-19 Radiography Dataset**:

🔗 **Kaggle:** https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset

### After downloading, place the dataset like this:

cnn_project/
Covid19-dataset/
train/
COVID19/
NORMAL/
PNEUMONIA/
test/
COVID19/
NORMAL/
PNEUMONIA/
validation/
COVID19/
NORMAL/
PNEUMONIA/

> ⚠️ Important:  
> The dataset is **ignored by .gitignore**, so users must download it manually.

---

## 🗂 Project Structure
```
cnn_project/
│
├── Covid19-dataset/           # Dataset (ignored by Git — download separately)
│   ├── train/
│   ├── test/
│   └── validation/
│
├── checkpoints/               # Trained CNN model weights (ignored)
├── results/                   # Metrics, plots, evaluation outputs
│
├── src/
│   ├── data.py                # Dataset loading + augmentations
│   ├── model.py               # Custom CNN implementation
│   ├── train.py               # Training loop
│   ├── eval.py                # Evaluation script
│   ├── inference.py           # Predict from a single image
│   ├── llm_wrapper.py         # Generates LLM-based explanations
│   └── __pycache__/
│
├── static/
│   ├── index.html             # Web UI for uploading & predicting images
│   ├── style.css              # Frontend styling
│   └── app.js                 # Browser-side calls to FastAPI
│
├── model_server.py            # FastAPI server exposing /predict
├── requirements.txt           # Python dependencies
├── LICENSE
└── README.md
```
---

## Installation

### 1️ Create virtual environment

  ```sh
  python -m venv venv
```
  Activate it:
  
   Windows
   ```
      venv\Scripts\activate
```
  ### macOS/Linux
  ```
  source venv/bin/activate
```
### 2 Install dependencies
   ```
  pip install -r requirements.txt
   ```
### Training:
 **Hyperparameters Used in the Report**
 ```# Runs used in the report:
python train.py --data_root ./Covid19-dataset --lr 2e-4  --batch_size 16 --epochs 30  --seed 123 
python train.py --data_root ./Covid19-dataset --lr 2e-4  --batch_size 16 --epochs 30  --seed 42
python train.py --data_root ./Covid19-dataset --lr 2e-4  --batch_size 16 --epochs 35  --seed 77   
```
**best parameters result**
```
 python train.py --data_root ./Covid19-dataset --lr 2e-4  --batch_size 16 --epochs 30  --seed 42  
```
    This script will: Load dataset , Train the CNN and Save best weights to:
    ```
            checkpoints/model_best.pth
      ```
### 3  Evaluation
```
  python src/eval.py --data_root ./Covid19-dataset --checkpoint ./checkpoints/last.pth --out_dir ./results --batch_size 16 --image_size 224       
```
**this where will the Output be saved**
```
    results/
```
### 4 Inference
  **Run prediction:**
       **Shows predicted label + probability distribution**
  ```
    python src/inference.py --image path/to/image.jpg
```
### 5 LLM Explanations
```
  python llm_client.py \
    --image path/to/image.jpg \
    --prediction COVID19   #Generate an explanation using the trained model output
```
### 6 FastAPI Server:
```
  uvicorn model_server:app --host 0.0.0.0 --port=8000 --reload #starting server
```
 ### opening the website via:
```
 http://localhost:8000
```


### License
MIT License — free for research, academic, and commercial use.
