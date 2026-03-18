from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__)
# ⚡ Autorise toutes les origines
CORS(app, resources={r"/*": {"origins": "*"}})

# HuggingFace client
client = InferenceClient(
    provider="hf-inference",  # ou fal-ai si tu as crédits
    api_key=os.environ.get("HF_TOKEN")
)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    size = data.get("size", "32x32")
    width, height = map(int, size.split("x"))

    image = client.text_to_image(
        prompt,
        model="stabilityai/stable-diffusion-xl-base-1.0",
        width=width,
        height=height
    )

    img_io = BytesIO()
    image.save(img_io, "PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
