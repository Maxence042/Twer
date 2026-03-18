from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__)
# ⚡ Autorise toutes les origines
CORS(app, resources={r"/*": {"origins": "*"}})

# ⚡ Config client Hugging Face
client = InferenceClient(
    provider="fal-ai",  # ou "hf-inference" si disponible pour ton modèle
    api_key=os.environ.get("HF_TOKEN")
)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    size = data.get("size", "32x32")  # format largeurxhauteur
    width, height = map(int, size.split("x"))

    # Génération image
    image = client.text_to_image(
        prompt,
        model="stabilityai/stable-diffusion-xl-base-1.0",
        width=width,
        height=height
    )

    # Retour PNG
    img_io = BytesIO()
    image.save(img_io, "PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
