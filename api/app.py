from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --------------------------------------------------
# FIX 3: Disable gradients globally (huge RAM saving)
# --------------------------------------------------
torch.set_grad_enabled(False)
# ----------------------------
# App & Device
# ----------------------------
app = Flask(__name__)
# --------------------------------------------------
# FIX 4: FORCE CPU (Render has no GPU)
# --------------------------------------------------
device = torch.device("cpu")

# --------------------------------------------------
# Load ResNet Model (Optimized)
# --------------------------------------------------
def load_resnet(model_path):
    #  Use ResNet18 instead of ResNet50 (critical)
    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )

    state_dict = torch.load(
        model_path,
        map_location="cpu"   # VERY IMPORTANT
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

MODEL_PATH = "saved_models/resnet_pneumonia.pth"
model = load_resnet(MODEL_PATH)

# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)

# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = request.files["image"].read()
    input_tensor = preprocess_image(image_bytes).to(device)

    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor)).item()

    prediction = "PNEUMONIA" if output > 0.5 else "NORMAL"

    return jsonify({
        "model": "ResNet18",
        "prediction": prediction,
        "confidence": round(output, 4)
    })

# --------------------------------------------------
# Run Server (Render-safe)
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)