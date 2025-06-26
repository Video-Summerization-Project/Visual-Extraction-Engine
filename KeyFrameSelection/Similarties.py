import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

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
    
    hasher = cv2.img_hash.PHash_create()
    seen_hashes = []
    distinct_records = []

    for frame, frame_idx in records:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_gray = cv2.resize(gray, (128, 128))

        img_hash = hasher.compute(frame)
        is_duplicate_hash = any(cv2.norm(img_hash, h, cv2.NORM_HAMMING) <= hash_threshold for h in seen_hashes)
        if is_duplicate_hash:
            continue
        seen_hashes.append(img_hash)

        is_distinct_ssim = True
        for prev_frame, _ in distinct_records[-ssim_compare_window:]:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            prev_resized = cv2.resize(prev_gray, (128, 128))
            if ssim(resized_gray, prev_resized) > ssim_threshold:
                is_distinct_ssim = False
                break

        if is_distinct_ssim:
            distinct_records.append((frame, frame_idx))

    return distinct_records

def _get_clip_embedding(frame):
    """
    Computes the CLIP image embedding for a given video frame.

    Args:
        frame (np.ndarray): A single video frame in BGR format (as returned by OpenCV).

    Returns:
        np.ndarray: A normalized 1D NumPy array representing the CLIP image embedding.
    """

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings[0].cpu().numpy()

def clip_filter(records, similarity_threshold=0.85, compare_window=5):
    """
    Filters frames using CLIP image embeddings and cosine similarity to remove semantically similar frames.

    Args:
        records (list): List of tuples (frame, frame_idx) representing sampled video frames.
        similarity_threshold (float, optional): Maximum cosine similarity allowed between embeddings to consider frames distinct. Defaults to 0.85.
        compare_window (int, optional): Number of recent embeddings to compare against for similarity. Defaults to 5.

    Returns:
        list: List of tuples (frame, frame_idx) representing filtered, semantically distinct keyframes.
    """

    distinct_records = []
    past_embeddings = []

    for frame, frame_idx in records:
        emb = _get_clip_embedding(frame)

        is_distinct = True
        for prev_emb in past_embeddings[-compare_window:]:
            sim = cosine_similarity([emb], [prev_emb])[0][0]
            if sim > similarity_threshold:
                is_distinct = False
                break

        if is_distinct:
            distinct_records.append((frame, frame_idx))
            past_embeddings.append(emb)

    return distinct_records
