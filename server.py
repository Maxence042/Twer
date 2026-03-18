from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os
from io import BytesIO
from PIL import Image

# ⚡ Création de l'app Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ⚡ Vérifie et récupère le token Hugging Face
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("⚠️ HF_TOKEN non défini dans les variables d'environnement !")
print("HF_TOKEN OK:", HF_TOKEN[:8], "...")  # juste 8 premiers caractères pour debug

# ⚡ Client SDXL via FAL
client = InferenceClient(
    provider="fal-ai",
    api_key=HF_TOKEN
)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json or {}
    prompt = data.get("prompt", "")
    size = data.get("size", "256x256")
    
    try:
        width, height = map(int, size.split("x"))
    except Exception:
        width = height = 256  # fallback si le format est incorrect

    # ⚡ Force la taille minimale pour SDXL
    if width < 256 or height < 256:
        width = height = 256

    try:
        # ⚡ Génération de l'image
        image = client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0",
            width=width,
            height=height
        )
    except Exception as e:
        # Retourne l'erreur exacte au frontend
        return jsonify({"error": str(e)}), 500

    # ⚡ Conversion en PNG pour le frontend
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
