import os
import uvicorn
import numpy as np
import cv2
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize FastAPI
app = FastAPI(title="FractureNet Intelligence API")

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
IMG_SIZE = (224, 224)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

# Global model state
model = None
model_loaded = False

def load_ai_model():
    """Attempts to load the best available trained model."""
    global model, model_loaded
    paths_to_try = [
        "best_fracturenet.keras",
        "../best_fracturenet.keras",
        "FracAtlas/best_fracturenet.keras",
        "c:/Users/HP/Downloads/archive (32)/best_fracturenet.keras"
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                # Load without compiling to avoid custom loss issues
                model = load_model(path, compile=False)
                model_loaded = True
                print(f"DEBUG: Model architecture loaded from {path}")
                print(f"DEBUG: Output shapes: {[o.shape for o in model.outputs]}")
                return
            except Exception as e:
                print(f"WARNING: Failed to load {path}: {e}")
    
    print("ERROR: No valid model found. Ensure 'best_fracturenet.keras' exists.")

@app.on_event("startup")
async def startup_event():
    load_ai_model()

def preprocess_image(img_bgr):
    """Resizes and normalizes the image for the DenseNet backbone."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, IMG_SIZE)
    img_norm = (img_res / 255.0 - MEAN) / STD
    return np.expand_dims(img_norm, axis=0).astype(np.float32)

def image_to_base64(img_bgr):
    """Encodes a BGR image to a base64 string for web display."""
    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/health")
async def health():
    return {"status": "active", "model_ready": model_loaded}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model_loaded:
        load_ai_model()
        if not model_loaded:
            raise HTTPException(status_code=503, detail="AI Model not initialized.")

    # Read and decode image
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_orig = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_orig is None: raise ValueError()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    H, W = img_orig.shape[:2]
    input_tensor = preprocess_image(img_orig)
    
    # Run Inference
    outputs = model.predict(input_tensor, verbose=0)
    
    # Handle dynamic output counts (Classification, Localization, optional Segmentation)
    if len(outputs) == 3:
        cp, lp, sp = outputs[0], outputs[1], outputs[2]
        has_mask = True
    elif len(outputs) == 2:
        cp, lp = outputs[0], outputs[1]
        sp = None
        has_mask = False
    else:
        # Fallback if it's just a single classification output
        cp = outputs if isinstance(outputs, np.ndarray) else outputs[0]
        lp = np.array([[0.5, 0.5, 0.1, 0.1]])
        has_mask = False

    prob = float(cp[0][0])
    is_fractured = prob > 0.5
    
    # Process Bounding Box
    xc, yc, bw, bh = lp[0]
    x1, y1 = int((xc - bw/2) * W), int((yc - bh/2) * H)
    x2, y2 = int((xc + bw/2) * W), int((yc + bh/2) * H)
    
    # Prepare Visualizations
    img_bbox = img_orig.copy()
    img_mask_overlay = img_orig.copy()

    if is_fractured:
        # Draw BBox (Cyan/Yellow)
        cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cv2.putText(img_bbox, f"DETECTION: {prob:.1%}", (max(0, x1), max(30, y1-10)), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 0), 2)

        # Process Mask if available
        if has_mask:
            mask_224 = sp[0, :, :, 0]
            mask_full = cv2.resize((mask_224 > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            overlay = np.zeros_like(img_orig)
            overlay[mask_full == 1] = [0, 0, 255] # Red highlight
            img_mask_overlay = cv2.addWeighted(img_orig, 0.7, overlay, 0.3, 0)
        else:
            # If no mask, just highlight the BBox area slightly
            overlay = np.zeros_like(img_orig)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            img_mask_overlay = cv2.addWeighted(img_orig, 0.85, overlay, 0.15, 0)

    return {
        "status": "Fractured" if is_fractured else "Normal",
        "probability": round(prob * 100, 1),
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "image_bbox": image_to_base64(img_bbox),
        "image_mask": image_to_base64(img_mask_overlay),
        "has_mask": has_mask
    }

# Serve static frontend
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
