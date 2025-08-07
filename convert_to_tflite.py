import tensorflow as tf

# Load your existing Keras model
model = tf.keras.models.load_model("model/model_3.keras")

# Convert to TensorFlow Lite with optimizations
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # size + speed optimizations
tflite_model = converter.convert()

# Save the converted model
with open("model/model_3.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model converted to model/model_3.tflite")
