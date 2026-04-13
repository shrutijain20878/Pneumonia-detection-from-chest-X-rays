from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os

# --------------------------------------------------
# Global settings
# --------------------------------------------------
torch.set_grad_enabled(False)   # Disable gradients
device = torch.device("cpu")   # CPU-only (Render safe)

# --------------------------------------------------
# Flask App
# --------------------------------------------------
app = Flask(__name__)

# --------------------------------------------------
# Load Trained ResNet18 Model
# --------------------------------------------------
def load_model(model_path):
    model = models.resnet18(weights=None)

    # Must match training architecture
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model
ROOT_DIR = os.getcwd()   # 🔥 points to project root on Render
MODEL_PATH = os.path.join(ROOT_DIR, "saved_models", "resnet_pneumonia.pth")
model = load_model(MODEL_PATH)

# MODEL_PATH = "saved_models/resnet_pneumonia.pth"
# model = load_model(MODEL_PATH)

# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
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
# Health Check
# --------------------------------------------------
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "Pneumonia Detection API is running"})

# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Image file is required"}), 400

    try:
        image_bytes = request.files["image"].read()
        input_tensor = preprocess_image(image_bytes).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probability = torch.sigmoid(logits).item()

        prediction = "PNEUMONIA" if probability >= 0.5 else "NORMAL"

        return jsonify({
            "model": "ResNet18",
            "prediction": prediction,
            "confidence": round(probability, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------
# Run Server
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)