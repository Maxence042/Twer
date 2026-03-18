import os
from io import BytesIO
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
from PIL import Image

# Initialisation Flask + CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Autorise toutes les origines

# Initialisation HuggingFace InferenceClient
client = InferenceClient(
    provider="hf-inference",  # ou "fal-ai" si tu veux
    api_key=os.environ.get("HF_TOKEN")  # Assure-toi que la variable d'env existe sur Railway
)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    size = data.get("size", "32x32")  # Format "WxH"
    width, height = map(int, size.split("x"))

    try:
        # Génération de l'image
        image = client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0",
            width=width,
            height=height
        )

        # Conversion en PNG
        img_io = BytesIO()
        image.save(img_io, "PNG")
        img_io.seek(0)
        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
