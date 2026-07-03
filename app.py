"""Microserviciu Flask optimizat pentru producția modelelor Keras multiclasă.

Rezolvă eroarea de conversie scalară prin indexarea corectă a axelor NumPy.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ACTUALIZARE: Modelul este căutat acum în folderul cizelat 'models'
MODEL_PATH = os.path.join(BASE_DIR, "models", "fruct_model_complet.keras")
model_instance = None

CLASS_LABELS = {0: "banane", 1: "mere", 2: "portocale"}


def get_production_model():
    """Încarcă leneș (lazy-load) modelul Keras în memorie la prima cerere."""
    global model_instance
    if model_instance is None:
        if not os.path.exists(MODEL_PATH):
            msg = f"Modelul '{MODEL_PATH}' nu există!"
            raise FileNotFoundError(msg)
        model_instance = load_model(MODEL_PATH)
        print("[PROD] Model incarcat cu succes in instanta Flask.")
    return model_instance


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint API care primește o imagine și returnează clasa detectată."""
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "Cheia 'image' lipseste"}), 400

    image_file = request.files["image"]

    try:
        # Preprocesare imagine uniformizată la 224x224
        img = Image.open(image_file).convert("RGB").resize((224, 224))
        img_array = img_to_array(img) / 255.0

        # Redimensionare rapidă la format batch (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)

        active_model = get_production_model()
        preds_batch = active_model.predict(img_array)

        # Extragem vectorul de predicție pentru prima imagine din batch
        predictions = preds_batch[0]

        # Calculăm indexul clasei dominante și scorul de confidență aferent
        predicted_index = int(np.argmax(predictions))
        confidence_score = float(predictions[predicted_index])

        return jsonify(
            {
                "status": "success",
                "predicted_class": CLASS_LABELS.get(predicted_index, "necunoscut"),
                "confidence": round(confidence_score, 4),
                "distribution": {
                    CLASS_LABELS[i]: float(prob) for i, prob in enumerate(predictions)
                },
            }
        ), 200

    except Exception as e:
        return jsonify({"status": "failed", "error_message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
