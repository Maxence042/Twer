# server.py
import os
from flask import Flask, request, send_file
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import replicate
import requests

app = Flask(__name__)
CORS(app)  # autorise toutes les origines

# Token Replicate depuis variable d'environnement
os.environ["REPLICATE_API_TOKEN"] = os.environ.get("REPLICATE_API_TOKEN", "")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    size = data.get("size", "64x64")  # format : "64x64", "128x128"
    width, height = map(int, size.split("x"))

    try:
        # Appel Replicate
        model = replicate.models.get("stability-ai/stable-diffusion-xl")
        output = model.predict(
            prompt=prompt,
            width=width,
            height=height,
            num_outputs=1
        )

        # Replicate retourne une URL de l'image
        image_url = output[0]
        resp = requests.get(image_url)
        img = Image.open(BytesIO(resp.content))

        # Conversion PNG → renvoi au frontend
        img_io = BytesIO()
        img.save(img_io, "PNG")
        img_io.seek(0)
        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        return {"error": str(e)}, 500

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
