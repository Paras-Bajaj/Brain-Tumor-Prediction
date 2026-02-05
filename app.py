from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os

# =========================
# CONFIG
# =========================
IMG_SIZE = 64
NUM_CLASSES = 4
CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

DEVICE = torch.device("cpu")

# =========================
# MODEL
# =========================
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(128 * 8 * 8, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES)
).to(DEVICE)

model.load_state_dict(torch.load("tumor_model.pth", map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =========================
# FLASK APP
# =========================
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template("templates/index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    image = Image.open(request.files['file']).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()

    return jsonify({'prediction': CLASSES[pred]})

# =========================
# RAILWAY ENTRY POINT
# =========================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
