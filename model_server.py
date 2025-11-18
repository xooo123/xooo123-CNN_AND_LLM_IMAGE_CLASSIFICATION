# model_server.py
import io
import os
import requests
from typing import Optional, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi import Form 
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import time

# Import the CNN model and LLM wrapper
from src.model import SimpleCNN
from src.llm_wrapper import explain_prediction

# ---------------------------
# Configuration
# ---------------------------
API_KEY = os.environ.get("MODEL_SERVER_API_KEY", "dev-key")

# Absolute path to checkpoint
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "best.pth")
print(f"[DEBUG] Looking for model at: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Checkpoint not found: {MODEL_PATH}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224  

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="COVID CNN Model Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Load model
# ---------------------------
def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    class_names = ckpt.get("class_names", ["Normal", "COVID"])
    model = SimpleCNN(num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, class_names

model, CLASS_NAMES = load_checkpoint(MODEL_PATH, DEVICE)

# ---------------------------
# Image preprocessing
# ---------------------------
def preprocess_image_bytes(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)  # 1,C,H,W

# ---------------------------
# Response model
# ---------------------------
class PredictionResponse(BaseModel):
    label: str
    score: float
    all_scores: Dict[str, float]
    explanation: Optional[str] = None

# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "classes": CLASS_NAMES}

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Query(None),
    use_llm: bool = Form(False),
    x_api_key: str = Header(...)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # Load or download image
    try:
        if file:
            content = await file.read()

            # ---- FIX START ----
            # Save uploaded file to disk
            os.makedirs("uploads", exist_ok=True)
            save_path = f"uploads/{int(time.time())}_{file.filename}"
            with open(save_path, "wb") as f_out:
                f_out.write(content)
            image_path = save_path  # <-- send REAL PATH to LLM
            # ---- FIX END ----

        elif image_url:
            r = requests.get(image_url, timeout=10)
            r.raise_for_status()
            content = r.content

            # Save downloaded image to disk
            os.makedirs("uploads", exist_ok=True)
            ext = image_url.split("?")[0].split(".")[-1]
            save_path = f"uploads/{int(time.time())}_remote.{ext}"
            with open(save_path, "wb") as f_out:
                f_out.write(content)
            image_path = save_path

        else:
            raise HTTPException(status_code=400, detail="Provide file or image_url")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    # Preprocess + inference
    try:
        x = preprocess_image_bytes(content).to(DEVICE)
        with torch.no_grad():
            out = model(x)
            probs = F.softmax(out, dim=1).cpu().numpy().tolist()[0]
            top_idx = int(out.argmax(1).cpu().item())
            label = CLASS_NAMES[top_idx]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    response = {
        "label": label,
        "score": float(probs[top_idx]),
        "all_scores": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))},
        "explanation": None
    }

    # LLM explanation
    if use_llm:
        try:
            explanation = explain_prediction(
                image_path=image_path,      # <-- SEND REAL SAVED FILE PATH
                predicted_label=label,
                probs=response["all_scores"]
            )
            response["explanation"] = explanation
        except Exception as e:
            response["explanation"] = f"(LLM explanation failed: {e})"

    return response


# ---------------------------
# Mount static files (MUST be LAST)
# ---------------------------
# This "catch-all" route must be defined AFTER all other API routes
# so that /predict, /health, etc. are handled by their functions first.
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ---------------------------
# Run server directly
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)