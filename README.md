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
Covid/
Normal/
Viral Pneumonia/
test/
Covid/
Normal/
Viral Pneumonia/
validation/
Covid/ (25 item)
Normal/(8 items exactly)
Viral Pneumonia/(19 item)

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
**on git bash**
  ```sh
python -m venv venv
```
 Activate it:
  
  Windows
   ```
venv/Scripts/activate

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
python src/train.py --data_root ./Covid19-dataset --lr 2e-4 --batch_size 16 --epochs 30 --seed 42
python src/train.py --data_root ./Covid19-dataset --lr 2e-4  --batch_size 16 --epochs 35  --seed 42
python src/train.py --data_root ./Covid19-dataset --lr 2e-4  --batch_size 16 --epochs 35  --seed 77
python src/train.py --data_root ./Covid19-dataset --lr 2e-4  --batch_size 16 --epochs 30  --seed 77
python src/train.py --data_root ./Covid19-dataset --lr 2e-4  --batch_size 16 --epochs 50  --seed 42
```
**best parameters result**
```
 python src/train.py --data_root ./Covid19-dataset --lr 2e-4  --batch_size 16 --epochs 30  --seed 42  

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
### 4 setting up the enviroment for LLM explanations
**4.1 get API key from openai website**
```
https://platform.openai.com/
```
**4.1.1 create an API key by going to Quickstart section and select ### create API key**

  **4.1.2 go to llm_client.py and src/llm_wrapper.py , find this section of the code and add your openai key save and run**
  ```
   api_key = ("OPENAI_API_KEY") #add your openai key here
  ```
**4.1.3 start ollama**
```
ollama serve
```
### 5 FastAPI Server:
```
 uvicorn model_server:app --reload  
```
 ### opening the website via:
```
 http://localhost:8000
```
## Note

##  Unit Tests

This project includes automated unit tests using **pytest** to verify:

- CNN forward pass correctness
- Output class dimensions
- Numerical stability (no NaNs)
- FastAPI server health endpoint
- LLM explanation fallback logic

To run all tests:

```bash
python -m pytest
```

**Keep in mind that openai might not work due to key issues and ollama is the local LLM in this project**

### License
MIT License — free for research, academic, and commercial use.
