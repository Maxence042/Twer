from flask import Flask, request, send_file
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os
from io import BytesIO
from PIL import Image

# ⚡ Crée l'application Flask
app = Flask(__name__)

# ⚡ Autorise toutes les origines, toutes les routes et méthodes
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ⚡ Token Hugging Face depuis variable d'environnement
client = InferenceClient(
    provider="fal-ai",
    api_key=os.environ.get("HF_TOKEN")
)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    size = data.get("size", "32x32")
    width, height = map(int, size.split("x"))

    # Génération image SDXL via Hugging Face
    image = client.text_to_image(
        prompt,
        model="stabilityai/stable-diffusion-xl-base-1.0",
        width=width,
        height=height
    )

    # Conversion PNG → envoi
    img_io = BytesIO()
    image.save(img_io, "PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
