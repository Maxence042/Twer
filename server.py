from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__)
# ⚡ Autorise toutes les origines pour fetch depuis n'importe quel site
CORS(app, resources={r"/*": {"origins": "*"}})

# Hugging Face token depuis variable d'environnement
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN n'est pas défini dans les variables d'environnement !")

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    size = data.get("size", "32x32")
    
    try:
        width, height = map(int, size.split("x"))
    except Exception as e:
        return jsonify({"status": "error", "message": f"Taille invalide : {size}"}), 400

    try:
        # Génération image
        image = client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0",
            width=width,
            height=height
        )
    except Exception as e:
        # Log de l'erreur dans Railway
        print("Erreur génération image :", e)
        return jsonify({"status": "error", "message": str(e)}), 500

    try:
        # Conversion PNG → envoi
        img_io = BytesIO()
        image.save(img_io, "PNG")
        img_io.seek(0)
        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        print("Erreur envoi image :", e)
        return jsonify({"status": "error", "message": "Impossible d'envoyer l'image"}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    print(f"Serveur démarré sur le port {port}")
    app.run(host="0.0.0.0", port=port)
