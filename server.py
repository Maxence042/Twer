from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import requests
from io import BytesIO

import replicate

app = Flask(__name__)
# ⚡ Autorise toutes les origines pour fetch depuis n'importe quel site
CORS(app, resources={r"/*": {"origins": "*"}})

# Récupération de la variable d'environnement
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN n'est pas défini !")

client = replicate.Client(api_token=REPLICATE_API_TOKEN)

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"status": "error", "message": "Aucun prompt fourni"}), 400

        # Appel Replicate
        print("Prompt reçu :", prompt)
        model = client.models.get("prunaai/z-image-turbo")  # Modèle Replicate
        output_urls = model.predict(prompt=prompt)

        print("Replicate output URLs:", output_urls)

        if not output_urls:
            return jsonify({"status": "error", "message": "Aucune URL générée par Replicate"}), 500

        # Téléchargement de l'image
        img_response = requests.get(output_urls[0])
        if img_response.status_code != 200:
            return jsonify({
                "status": "error",
                "message": f"Impossible de récupérer l'image ({img_response.status_code})"
            }), 500

        img_io = BytesIO(img_response.content)
        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        # DEBUG : retourne l'erreur exacte dans JSON
        print("ERREUR :", e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
