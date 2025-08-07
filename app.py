import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from pyngrok import ngrok

# Initialize app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Load model
MODEL_PATH = "model/model_3.keras"
model = load_model(MODEL_PATH)

# Class names
class_names = [
    "butterfly",
    "cat",
    "chicken",
    "cow",
    "dog",
    "elephant",
    "horse",
    "sheep",
    "spider",
    "squirrel",
]


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((64, 64))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0

        prediction = model.predict(image)[0]
        top_index = int(np.argmax(prediction))
        result = {
            "prediction": class_names[top_index],
            "confidence": float(prediction[top_index]),
            "all_probs": {
                class_names[i]: float(prediction[i]) for i in range(len(class_names))
            },
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Ngrok for Colab
if __name__ == "__main__":
    ngrok.set_auth_token(
        "30xvzkKd16GEExWAu01vun8HLJa_39YSAXFrC1eztGYdERs8d"
    )  # Replace with your token
    port = 5000
    public_url = ngrok.connect(port)
    print(f"🚀 App is live at: {public_url}")
    app.run(port=port)
