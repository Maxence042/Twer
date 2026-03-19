import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate

app = Flask(__name__)
CORS(app)  # Autorise toutes les origines pour fetch depuis ton frontend

# Récupère le token Replicate depuis les variables d'environnement
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN n'est pas défini dans les variables d'environnement !")

client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Route pour générer l'image
@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    size = data.get("size", "512x512")  # Taille par défaut
    
    # On prend un modèle Stable Diffusion disponible sur Replicate
    model = client.models.get("stability-ai/stable-diffusion")
    
    # Predict renvoie une URL de l'image générée
    output = model.predict(prompt=prompt, width=int(size.split("x")[0]), height=int(size.split("x")[1]))
    
    return jsonify({"image_url": output[0]})

# Route health
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
