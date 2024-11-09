from flask import Flask, request, render_template, send_from_directory
import os
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


FRAMES_DIR = "E:\\THIHE\\testfitty one\\SegmentVideo\\seg1\\SegmentVideo"
EMBEDDINGS_FILE = "E:\\Đồ án chuyên ngành\\source test\\embedding\\image_embeddings.npy"


embeddings = np.load(EMBEDDINGS_FILE)


app = Flask(__name__)


@app.route('/frames/<path:filename>')
def serve_frame(filename):
    return send_from_directory(FRAMES_DIR, filename)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query")
        top_k = request.form.get("top_k", 50)  
        try:
            top_k = int(float(top_k))  
        except ValueError:
            top_k = 50  

        total_frames = len([f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        top_k = min(max(1, top_k), total_frames)  

        top_frames = search_top_frames(query, top_k)
        return render_template("index.html", query=query, top_frames=top_frames)
    return render_template("index.html", query=None, top_frames=[])

def search_top_frames(query, top_k):
    
    text_input = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input).cpu().numpy()
    
    similarities = np.dot(embeddings, text_features.T).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]  

    all_files = [f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return [all_files[i] for i in top_indices]

if __name__ == "__main__":
    app.run(debug=True)
