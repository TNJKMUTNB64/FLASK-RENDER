from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# โหลดโมเดล Keras
model = tf.keras.models.load_model("fall_detection_model_AccGyr.keras")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("inputs")
    if not data or len(data) != 6:
        return jsonify({"error": "Invalid input format. Expecting 6 values."}), 400

    # เติมลำดับเวลา
    time_steps = 50
    input_sequence = np.tile(data, (time_steps, 1))
    input_array = input_sequence.reshape(1, time_steps, 6)

    # ทำการพยากรณ์
    prediction = model.predict(input_array)
    result = "Fall" if prediction[0][0] > 0.75 else "No Fall"

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
