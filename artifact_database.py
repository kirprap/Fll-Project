"""
AI-powered artifact analysis (local-only, ViT)
Uses:
 - transformers + torch for ViT image classification
 - transformers CLIPModel + CLIPProcessor for image/text embeddings
 - PIL / numpy for image handling and numeric ops

Assumes required packages are installed:
pip install transformers torch pillow numpy
"""

from typing import List, Dict, Optional, Any, Tuple
import base64
import io
import logging

# Core ML libs
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification, CLIPProcessor, CLIPModel

# Simple logger
logger = logging.getLogger(__name__)

# Constants / models
_VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
_CLIP_MODEL = "openai/clip-vit-base-patch32"

# Lego heuristics
_LEGO_COLORS = ["red", "blue", "yellow", "green", "white", "black"]
_LEGO_TYPES = [
    "LEGO brick 2x4",
    "LEGO plate 1x2",
    "LEGO tile 1x1",
    "LEGO slope 2x2",
    "LEGO minifigure head",
    "LEGO minifigure torso",
    "LEGO technic pin",
    "LEGO brick 1x1",
    "LEGO brick 2x2"
]
_LEGO_CONFIDENCE_THRESHOLD = 0.70

# Model caches
_VIT_PROCESSOR: Optional[AutoImageProcessor] = None
_VIT_MODEL: Optional[ViTForImageClassification] = None

_CLIP_PROCESSOR: Optional[CLIPProcessor] = None
_CLIP_MODEL_OBJ: Optional[CLIPModel] = None


def _load_device() -> torch.device:
    """Prefer GPU if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_vit():
    """Load and cache ViT processor+model for classification."""
    global _VIT_PROCESSOR, _VIT_MODEL
    if _VIT_MODEL is None or _VIT_PROCESSOR is None:
        logger.info(f"Loading ViT model: {_VIT_MODEL_NAME}")
        _VIT_PROCESSOR = AutoImageProcessor.from_pretrained(_VIT_MODEL_NAME)
        _VIT_MODEL = ViTForImageClassification.from_pretrained(_VIT_MODEL_NAME)
        device = _load_device()
        _VIT_MODEL.to(device)
        _VIT_MODEL.eval()
        # Enable inference mode for better performance
        torch.set_grad_enabled(False)
        logger.info(f"ViT model loaded successfully on {device}")
    return _VIT_PROCESSOR, _VIT_MODEL


def _load_clip():
    """Load and cache CLIP processor+model for embeddings."""
    global _CLIP_PROCESSOR, _CLIP_MODEL_OBJ
    if _CLIP_MODEL_OBJ is None or _CLIP_PROCESSOR is None:
        logger.info(f"Loading CLIP model: {_CLIP_MODEL}")
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(_CLIP_MODEL)
        _CLIP_MODEL_OBJ = CLIPModel.from_pretrained(_CLIP_MODEL)
        device = _load_device()
        _CLIP_MODEL_OBJ.to(device)
        _CLIP_MODEL_OBJ.eval()
        # Enable inference mode for better performance
        torch.set_grad_enabled(False)
        logger.info(f"CLIP model loaded successfully on {device}")
    return _CLIP_PROCESSOR, _CLIP_MODEL_OBJ


def _decode_base64_image_to_pil(base64_image: str) -> Image.Image:
    """Decode base64 string to a PIL Image (RGB)."""
    try:
        img_bytes = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}") from e


def _vit_classify_image(pil_img: Image.Image) -> Tuple[str, float]:
    """Run ViT classification locally and return (label, confidence)."""
    processor, model = _load_vit()
    device = _load_device()
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        conf, idx = probs.max(-1)
        label = model.config.id2label.get(int(idx.item()), str(idx.item()))
        return label, float(conf.item())


def _clip_image_embedding(pil_img: Image.Image) -> np.ndarray:
    processor, model = _load_clip()
    device = _load_device()
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        img_features = model.get_image_features(**inputs)
    emb = img_features.cpu().numpy().ravel()
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def _clip_text_embeddings(texts: List[str]) -> List[np.ndarray]:
    processor, model = _load_clip()
    device = _load_device()
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        txt_feats = model.get_text_features(**inputs)
    arr = txt_feats.cpu().numpy()
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    return [row.ravel() for row in arr]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def analyze_artifact_image(base64_image: str) -> Dict[str, Any]:
    """Analyze artifact image using ViT and CLIP heuristics."""
    try:
        pil_img = _decode_base64_image_to_pil(base64_image)
        try:
            label, confidence = _vit_classify_image(pil_img)
        except Exception:
            logger.exception("ViT classification failed")
            label, confidence = "unknown", 0.0

        analysis = {
            'name': label,
            'value': "Estimated value requires expert appraisal",
            'age': "Age estimation requires expert analysis",
            'description': f"This image is classified as \"{label}\".",
            'cultural_context': "Cultural context requires expert input",
            'confidence': float(confidence),
            'material': "Material analysis requires physical inspection",
            'function': "Function inferred from type; verify manually",
            'rarity': "Rarity assessment requires comparison"
        }

        # LEGO detection
        try:
            lego_info = identify_lego_brick(base64_image)
            if lego_info and lego_info.get('is_lego') and lego_info.get('best_similarity', 0.0) >= _LEGO_CONFIDENCE_THRESHOLD:
                best_label = lego_info['predictions'][0]['label']
                best_sim = float(lego_info['predictions'][0]['similarity'])
                analysis.update({
                    'name': f"LEGO: {best_label}",
                    'description': f"Detected LEGO component: {best_label} (similarity {best_sim:.3f}). Secondary: {[p['label'] for p in lego_info['predictions'][1:3]]}",
                    'confidence': float(max(confidence, best_sim)),
                    'material': "ABS plastic",
                    'function': "Toy / construction element",
                    'rarity': "Common (verify collector variants)",
                    'value': "Typical retail/collector value"
                })
        except Exception:
            logger.exception("LEGO detection failed")
        return analysis

    except Exception as e:
        logger.exception("Failed to analyze artifact image")
        raise RuntimeError(f"Failed to analyze artifact image: {e}") from e


def identify_lego_brick(base64_image: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
    """Identify LEGO parts by CLIP similarity."""
    try:
        pil_img = _decode_base64_image_to_pil(base64_image)
        img_emb = _clip_image_embedding(pil_img)

        candidate_texts = []
        for t in _LEGO_TYPES:
            candidate_texts.append(t)
            for c in _LEGO_COLORS:
                candidate_texts.extend([f"{c} {t}", f"{t} {c}"])
        # dedupe
        candidate_texts = list(dict.fromkeys(candidate_texts))

        text_embs = _clip_text_embeddings(candidate_texts)

        sims = [(lbl, _cosine_similarity(img_emb, emb)) for lbl, emb in zip(candidate_texts, text_embs)]
        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[:top_k]
        predictions = [{'label': lbl, 'similarity': float(sim)} for lbl, sim in top]
        best_similarity = float(predictions[0]['similarity']) if predictions else 0.0
        best_label = predictions[0]['label'] if predictions else ""
        is_lego = ('lego' in best_label.lower() or 'brick' in best_label.lower() or 'minifigure' in best_label.lower()) and best_similarity >= 0.45

        return {'is_lego': bool(is_lego), 'predictions': predictions, 'best_similarity': best_similarity}

    except Exception:
        logger.exception("identify_lego_brick error")
        return None


# compare_with_reference and other functions remain unchanged
# (they call CLIP embeddings, not ViT)
