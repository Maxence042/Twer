from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os
from io import BytesIO
from PIL import Image

# ⚡ Crée l'application Flask
app = Flask(__name__)

# ⚡ Autorise toutes les origines, toutes les routes et méthodes
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ⚡ Récupère le token Hugging Face depuis les variables d'environnement
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("⚠️ HF_TOKEN non défini dans les variables d'environnement !")

client = InferenceClient(
    provider="fal-ai",
    api_key=HF_TOKEN
)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json or {}
    prompt = data.get("prompt", "")
    size = data.get("size", "128x128")  # taille par défaut
    width, height = map(int, size.split("x"))

    # ⚠️ Force la taille minimale pour SDXL
    if width < 128 or height < 128:
        width = height = 128

    try:
        # Génération image SDXL via Hugging Face
        image = client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0",
            width=width,
            height=height
        )
    except Exception as e:
        # Retourne l'erreur au frontend
        return jsonify({"error": str(e)}), 500

    # Conversion PNG → envoi
    img_io = BytesIO()
    image.save(img_io, "PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
