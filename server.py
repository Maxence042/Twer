from flask import Flask, request, send_file
from huggingface_hub import InferenceClient
import os
from io import BytesIO

app = Flask(__name__)

# 🔑 Mets ton token ici temporairement (on sécurisera après)
HF_TOKEN = "hf_XXXXXXXXXXXX"

client = InferenceClient(
    provider="fal-ai",
    api_key=HF_TOKEN,
)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")

    image = client.text_to_image(
        prompt,
        model="stabilityai/stable-diffusion-xl-base-1.0",
    )

    img_io = BytesIO()
    image.save(img_io, "PNG")
    img_io.seek(0)

    return send_file(img_io, mimetype="image/png")

if __name__ == "__main__":
    app.run(port=3000)