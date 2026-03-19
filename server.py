import os
import replicate
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from io import BytesIO
import requests

app = Flask(__name__)
# ⚡ Autorise toutes les origines pour fetch depuis n'importe quel site
CORS(app, resources={r"/*": {"origins": "*"}})

# 🔑 Ton token Replicate depuis les variables d'environnement
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN n'est pas défini !")

client = replicate.Client(api_token=REPLICATE_API_TOKEN)

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt", "pixel art dragon")

        # Appel du modèle prunaai/z-image-turbo
        output_urls = client.models.get("prunaai/z-image-turbo").predict(
            prompt=prompt
        )

        # Télécharge la première image générée
        img_data = requests.get(output_urls[0]).content
        img_io = BytesIO(img_data)
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
