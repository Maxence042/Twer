import os
import requests
from flask import Flask, request, send_file
from flask_cors import CORS
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)  # autorise toutes les requêtes cross-origin

# ⚡ Replicate API token
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")  # à définir dans Railway

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    size = data.get("size", "64x64")
    width, height = map(int, size.split("x"))

    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "version": "5d94e7d07f45c843eb14a9070a8de9f3b3edb05e7e0d7b2120a0a3b191c68f84",  # SDXL Replicate version
        "input": {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_outputs": 1
        }
    }

    # Appel à l’API Replicate
    response = requests.post("https://api.replicate.com/v1/predictions", headers=headers, json=payload)
    if response.status_code != 201 and response.status_code != 200:
        return {"error": response.text}, 500

    output_url = response.json()["output"][0]

    # Récupérer l’image
    resp = requests.get(output_url)
    img = Image.open(BytesIO(resp.content))

    # Retourner PNG
    img_io = BytesIO()
    img.save(img_io, "PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
