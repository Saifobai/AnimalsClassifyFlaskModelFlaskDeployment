import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage

import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Path to model
MODEL_PATH = "model/model_3.keras"

# Load model once at startup with compile=False to reduce memory use
model = load_model(MODEL_PATH, compile=False)

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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        # Process the image
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((64, 64))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0) / 255.0

        # Run prediction
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
