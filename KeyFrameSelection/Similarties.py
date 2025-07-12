import cv2
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from concurrent.futures import ThreadPoolExecutor

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", trust_remote_code=True, use_safetensors=True)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

def _resize_gray(frame):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (128, 128))

def hash_filter(records, hash_threshold=5, ssim_threshold=0.90, ssim_compare_window=3):
    """
    Filters out visually similar frames using perceptual hashing and SSIM.

    Args:
        records (list): List of tuples (frame, frame_idx) representing sampled video frames.
        hash_threshold (int, optional): Maximum Hamming distance between perceptual hashes to consider frames as duplicates. Defaults to 5.
        ssim_threshold (float, optional): Maximum SSIM score to consider frames as distinct. Defaults to 0.90.
        ssim_compare_window (int, optional): Number of most recent accepted frames to compare against using SSIM. Defaults to 3.

    Returns:
        list: List of tuples (frame, frame_idx) representing filtered, distinct keyframes.
    """
    resized_cache = {idx: _resize_gray(frame) for frame, idx in records}

    def compute_hash(frame):
        return imagehash.phash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    with ThreadPoolExecutor() as executor:
        hashes = list(executor.map(lambda x: compute_hash(x[0]), records))

    seen_hashes = []
    distinct = []

    for i, (frame, frame_idx) in enumerate(records):
        img_hash = hashes[i]

        if any(abs(img_hash - h) <= hash_threshold for h in seen_hashes):
            continue
        seen_hashes.append(img_hash)

        is_distinct = True
        resized_gray = resized_cache[frame_idx]
        for _, prev_idx in distinct[-ssim_compare_window:]:
            prev_gray = resized_cache[prev_idx]
            if ssim(resized_gray, prev_gray) > ssim_threshold:
                is_distinct = False
                break

        if is_distinct:
            distinct.append((frame, frame_idx))

    return distinct

def _get_clip_embeddings(frames):
    """
    Computes the CLIP image embedding for a given video frame.

    Args:
        frame (np.ndarray): A single video frame in BGR format (as returned by OpenCV).

    Returns:
        np.ndarray: A normalized 1D NumPy array representing the CLIP image embedding.
    """
    images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    inputs = clip_processor(images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        normed = torch.nn.functional.normalize(features, p=2, dim=1)
    return normed.cpu().numpy()

def clip_filter(records, similarity_threshold=0.85, compare_window=5, batch_size=8):
    """
    Filters frames using CLIP embeddings and cosine similarity in batch mode (CPU-optimized).

    Args:
        records (list): List of (frame, frame_idx) tuples.
        similarity_threshold (float): Max cosine similarity to keep frame distinct.
        compare_window (int): How many past frames to compare against.
        batch_size (int): Number of frames to embed per batch (for speed).

    Returns:
        list: Filtered list of (frame, frame_idx) tuples with distinct content.
    """
    frames, frame_idxs = zip(*records)
    embeddings = []

    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        batch_embs = _get_clip_embeddings(batch_frames)
        embeddings.extend(batch_embs)

    distinct = []
    past_embeddings = []

    for i, emb in enumerate(embeddings):
        is_distinct = True
        for prev_emb in past_embeddings[-compare_window:]:
            sim = cosine_similarity([emb], [prev_emb])[0][0]
            if sim > similarity_threshold:
                is_distinct = False
                break

        if is_distinct:
            distinct.append((frames[i], frame_idxs[i]))
            past_embeddings.append(emb)

    return distinct
