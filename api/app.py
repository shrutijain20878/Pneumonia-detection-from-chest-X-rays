from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ----------------------------
# App & Device
# ----------------------------
app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load ResNet Model
# ----------------------------
def load_resnet(model_path):
    model = models.resnet50(weights=None)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    return model


MODEL_PATH = "saved_models/resnet_pneumonia.pth"
model = load_resnet(MODEL_PATH)

# ----------------------------
# Image Preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)

# ----------------------------
# Prediction Endpoint
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    temp_path = "temp.jpg"
    image_file.save(temp_path)

    input_tensor = preprocess_image(temp_path).to(device)

    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor)).item()

    os.remove(temp_path)

    prediction = "PNEUMONIA" if output > 0.5 else "NORMAL"

    return jsonify({
        "model": "ResNet50",
        "prediction": prediction,
        "confidence": round(output, 4)
    })


# ----------------------------
# Run Server
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
