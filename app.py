from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
import torch
import io
import requests
# load model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Flask app
app = Flask(__name__)

def generateCaption(prompt, img):
    # downscale resolution
    inputs = processor(img, prompt, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


@app.route('/', methods=['POST', 'GET']) # route??
def index():
    if request.method == 'POST':

        try:
            file = request.files["image"]  # argument has name
            image = Image.open(file.stream)
            text_input = request.form['prompt']
            caption = generateCaption(text_input, image)
            # Convert the image to JPEG format and save it to a bytes buffer
            return render_template('update.html' , caption=caption)
        except:
            return 'Wrong file type uploaded'
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)