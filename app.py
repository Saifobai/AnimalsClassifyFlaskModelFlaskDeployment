import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model/model_3.tflite")
interpreter.allocate_tensors()

# Get input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
        image = np.expand_dims(np.array(image) / 255.0, axis=0).astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]["index"], image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]["index"])[0]

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
