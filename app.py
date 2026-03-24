from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Load model with error handling
try:
    if not os.path.exists("tumor_model.pth"):
        logger.error("Model file 'tumor_model.pth' not found!")
        raise FileNotFoundError("Model file not found")
    
    model.load_state_dict(torch.load("tumor_model.pth", map_location=DEVICE))
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

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
    try:
        return render_template("index.html")
    except Exception as e:
        logger.error(f"Template error: {e}")
        return jsonify({'error': 'Template not found'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for Railway"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model is loaded
    if model is None:
        return jsonify({'error': 'Model not loaded. Please contact administrator.'}), 503
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        # Open and process image
        image = Image.open(file).convert('RGB')
        
        # Transform and predict
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            confidence = probabilities[0][pred].item() * 100
        
        return jsonify({
            'prediction': CLASSES[pred],
            'confidence': round(confidence, 2),
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 500

# =========================
# RAILWAY ENTRY POINT
# =========================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)