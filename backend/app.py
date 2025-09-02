import os
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS

from models.waste_cnn import WasteCNN
from utils.preprocess import load_image, pil_to_tensor, preprocess_for_cnn

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="/")
CORS(app)  # enable if you serve frontend on a different origin

# ---- Mock auth storage (prototype only) ----
ACTIVE_TOKENS = set()

# ---- Load model wrapper ----
MODEL = WasteCNN(
    label_map_path=str(BASE_DIR / "backend" / "models" / "label_map.json"),
    weights_path=str(BASE_DIR / "data" / "waste_cnn.pt")
)

# ---- Basic waste category mapping from object label -> waste type ----
OBJECT_TO_WASTE = {
    "plastic_bottle": "plastic",
    "aluminum_can": "metal", 
    "cardboard": "paper",
    "paper": "paper",
    "glass_jar": "glass",
    "food_waste": "organic",
    "old_phone": "e-waste",
    "laptop": "e-waste",
    "syringe": "medical",
    "battery": "e-waste",
    "other": "other"
}

# ---- Heuristic recyclability percentages per waste type ----
RECYCLABILITY = {
    "plastic": 40,
    "metal": 75,
    "paper": 65,
    "glass": 85,
    "organic": 0,   # compostable but not "recyclable"
    "e-waste": 30,  # varies widely
    "medical": 5,   # typically hazardous; specialized disposal
    "other": 10
}

@app.route("/")
def root():
    # redirect to login page
    return redirect("/index.html")

# -------- Static frontend files --------
@app.route("/<path:path>")
def serve_frontend(path):
    # Serve static files from /frontend
    full = FRONTEND_DIR / path
    if full.exists():
        return send_from_directory(str(FRONTEND_DIR), path)
    return "Not Found", 404

# -------- API: login (prototype) --------
@app.post("/api/login")
def api_login():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    # Prototype: accept anything non-empty
    if username and password:
        token = f"token_{username}"
        ACTIVE_TOKENS.add(token)
        return jsonify({"ok": True, "token": token})
    return jsonify({"ok": False, "error": "Invalid credentials"}), 401

def _auth_ok(req) -> bool:
    token = req.headers.get("Authorization", "").replace("Bearer ", "")
    return token in ACTIVE_TOKENS

# -------- API: predict --------
@app.post("/api/predict")
def api_predict():
    if not _auth_ok(request):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image uploaded with key 'image'"}), 400

    img_file = request.files["image"]
    try:
        # Load and preprocess image
        pil = load_image(img_file)
        processed_img = preprocess_for_cnn(pil)
        
        # Get prediction from CNN model
        idx, confidence = MODEL.predict_image(processed_img)
        obj_label = MODEL.idx_to_label(idx)
        waste_type = OBJECT_TO_WASTE.get(obj_label, "other")
        recyclability = RECYCLABILITY.get(waste_type, 10)

        # Get model info for debugging
        model_info = MODEL.get_model_info()

        return jsonify({
            "ok": True,
            "object_label": obj_label,
            "waste_type": waste_type,
            "confidence": round(float(confidence), 3),
            "recyclability_percent": recyclability,
            "model_status": model_info["status"],
            "device": model_info["device"]
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction error: {error_details}")
        return jsonify({"ok": False, "error": str(e)}), 500

# -------- API: model info --------
@app.get("/api/model-info")
def api_model_info():
    """Get information about the loaded model"""
    if not _auth_ok(request):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    
    model_info = MODEL.get_model_info()
    return jsonify({"ok": True, **model_info})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
