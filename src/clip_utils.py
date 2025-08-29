# src/clip_utils.py
import os
import numpy as np
from typing import List, Tuple
from transformers import CLIPProcessor, CLIPModel
import torch
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"

# initialize model and processor once
_clip_model = None
_clip_processor = None

def init_clip(model_name: str = None):
    global _clip_model, _clip_processor
    model_name = model_name or CLIP_MODEL
    if _clip_model is None or _clip_processor is None:
        _clip_model = CLIPModel.from_pretrained(model_name).to(device)
        _clip_processor = CLIPProcessor.from_pretrained(model_name)
    return _clip_model, _clip_processor

def embed_frames(frames: List[Tuple[float, "PIL.Image.Image"]], batch_size: int = 32) -> Tuple[np.ndarray, List[float]]:
    """
    frames: list of (timestamp, PIL.Image)
    Returns embeddings and timestamps.
    """
    model, processor = init_clip()
    timestamps = [ts for ts, _ in frames]
    imgs = [img for _, img in frames]
    all_embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            batch_imgs = imgs[i:i+batch_size]
            inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
            outputs = model.get_image_features(**inputs)
            emb = outputs.cpu().numpy()
            all_embs.append(emb)
    all_embs = np.vstack(all_embs)
    # normalize
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-12
    all_embs = all_embs / norms
    return all_embs, timestamps

def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 8) -> np.ndarray:
    """Simple KMeans clustering on normalized embeddings. Returns cluster labels."""
    n = min(len(embeddings), max(1, int(n_clusters)))
    if len(embeddings) <= n:
        # trivial labels
        return np.arange(len(embeddings))
    km = KMeans(n_clusters=n, random_state=42, n_init="auto")
    labels = km.fit_predict(embeddings)
    return labels

def pick_representative_frames(embeddings: np.ndarray, frames: List[Tuple[float, "PIL.Image.Image"]], labels: np.ndarray, top_k: int = 1):
    """
    For each cluster, pick frame(s) closest to centroid.
    Returns list of dict {cluster, timestamp, image, distance}
    """
    results = []
    unique_labels = sorted(set(labels))
    for lab in unique_labels:
        idxs = np.where(labels == lab)[0]
        if len(idxs) == 0:
            continue
        cluster_embs = embeddings[idxs]
        centroid = cluster_embs.mean(axis=0, keepdims=True)
        # compute distances (cosine since embeddings normalized => higher dot means closer)
        dots = (cluster_embs @ centroid.T).squeeze()
        order = np.argsort(-dots)  # descending
        for r in range(min(top_k, len(order))):
            idx = idxs[order[r]]
            ts, img = frames[idx]
            results.append({"cluster": lab, "timestamp": ts, "image": img, "score": float(dots[order[r]])})
    # sort by cluster label
    results = sorted(results, key=lambda x: x["cluster"])
    return results
