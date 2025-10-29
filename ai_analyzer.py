"""
AI-powered artifact analysis (local-only)
Uses:
 - transformers + torch for ResNet image classification
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
from transformers import AutoImageProcessor, ResNetForImageClassification, CLIPProcessor, CLIPModel

# Simple logger
logger = logging.getLogger(__name__)

# Constants / models
_VIT_MODEL = "microsoft/resnet-50"
# Using a CLIP checkpoint available through transformers
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
_RESNET_PROCESSOR: Optional[AutoImageProcessor] = None
_RESNET_MODEL: Optional[ResNetForImageClassification] = None

_CLIP_PROCESSOR: Optional[CLIPProcessor] = None
_CLIP_MODEL_OBJ: Optional[CLIPModel] = None


def _load_resnet_device() -> torch.device:
    """Prefer GPU if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_resnet():
    """Load and cache ResNet processor+model for classification."""
    global _RESNET_PROCESSOR, _RESNET_MODEL
    if _RESNET_MODEL is None or _RESNET_PROCESSOR is None:
        _RESNET_PROCESSOR = AutoImageProcessor.from_pretrained(_VIT_MODEL)
        _RESNET_MODEL = ResNetForImageClassification.from_pretrained(_VIT_MODEL)
        # move to device if available
        device = _load_resnet_device()
        _RESNET_MODEL.to(device)
        _RESNET_MODEL.eval()
    return _RESNET_PROCESSOR, _RESNET_MODEL


def _load_clip():
    """Load and cache CLIP processor+model for embeddings."""
    global _CLIP_PROCESSOR, _CLIP_MODEL_OBJ
    if _CLIP_MODEL_OBJ is None or _CLIP_PROCESSOR is None:
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(_CLIP_MODEL)
        _CLIP_MODEL_OBJ = CLIPModel.from_pretrained(_CLIP_MODEL)
        # put model on device if available
        device = _load_resnet_device()
        _CLIP_MODEL_OBJ.to(device)
        _CLIP_MODEL_OBJ.eval()
    return _CLIP_PROCESSOR, _CLIP_MODEL_OBJ


def _decode_base64_image_to_pil(base64_image: str) -> Image.Image:
    """Decode base64 string to a PIL Image (RGB). Raises ValueError on invalid input."""
    try:
        img_bytes = base64.b64decode(base64_image)
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}") from e
    if not img_bytes:
        raise ValueError("Decoded image bytes are empty")
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Unable to parse image bytes as an image: {e}") from e
    return img


def _resnet_classify_image(pil_img: Image.Image) -> Tuple[str, float]:
    """Run ResNet classification locally and return (label, confidence)."""
    processor, model = _load_resnet()
    device = _load_resnet_device()
    inputs = processor(images=pil_img, return_tensors="pt")
    # move tensors to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape (1, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        conf, idx = probs.max(-1)
        conf_val = float(conf.item())
        idx_val = int(idx.item())
        label = model.config.id2label.get(idx_val, str(idx_val))
        return label, conf_val


def _clip_image_embedding(pil_img: Image.Image) -> np.ndarray:
    """Return a 1-D numpy array image embedding from CLIP (normalized)."""
    processor, model = _load_clip()
    device = _load_resnet_device()
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        img_features = model.get_image_features(**inputs)  # tensor (1, dim)
    emb = img_features.cpu().numpy().ravel().astype(float)
    # normalize to unit vector to simplify cosine computations
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def _clip_text_embeddings(texts: List[str]) -> List[np.ndarray]:
    """Return a list of 1-D numpy arrays for the input texts using CLIP text encoder."""
    processor, model = _load_clip()
    device = _load_resnet_device()
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        txt_feats = model.get_text_features(**inputs)  # (batch, dim)
    arr = txt_feats.cpu().numpy().astype(float)
    # normalize each row
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    return [row.ravel() for row in arr]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D numpy arrays (assumes normalization optional)."""
    if a is None or b is None:
        return -1.0
    # If not normalized, compute via dot/(||a||*||b||)
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return -1.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    sim = float(np.dot(a, b) / denom)
    # clamp numeric issues
    return float(max(-1.0, min(1.0, sim)))


def analyze_artifact_image(base64_image: str) -> Dict[str, Any]:
    """Analyze artifact image using local ResNet classifier and CLIP-based heuristics."""
    try:
        pil_img = _decode_base64_image_to_pil(base64_image)

        # classification
        try:
            label, confidence = _resnet_classify_image(pil_img)
        except Exception as e:
            logger.exception("ResNet classification failed")
            label, confidence = "unknown", 0.0

        # base analysis skeleton
        analysis = {
            'name': label,
            'value': "Estimated value requires expert appraisal",
            'age': "Age estimation requires detailed material/stratigraphic analysis",
            'description': f"This image is classified as \"{label}\".",
            'cultural_context': "Cultural context requires expert analysis and provenance data",
            'confidence': float(confidence),
            'material': "Material analysis requires physical or spectral inspection",
            'function': "Function inferred from type; needs expert confirmation",
            'rarity': "Rarity assessment requires provenance and collection comparison"
        }

        # Try lego detection using CLIP heuristics
        try:
            lego_info = identify_lego_brick(base64_image, top_k=5)
            if lego_info and lego_info.get('is_lego') and lego_info.get('best_similarity', 0.0) >= _LEGO_CONFIDENCE_THRESHOLD:
                best_label = lego_info['predictions'][0]['label']
                best_sim = float(lego_info['predictions'][0]['similarity'])
                analysis.update({
                    'name': f"LEGO: {best_label}",
                    'description': f"Detected LEGO component: {best_label} (similarity {best_sim:.3f}). Secondary: {[p['label'] for p in lego_info['predictions'][1:3]]}",
                    'confidence': float(max(confidence, best_sim)),
                    'material': "ABS plastic (typical for LEGO bricks)",
                    'function': "Toy / construction element",
                    'rarity': "Common (verify special collector variants separately)",
                    'value': "Typical retail/collector value; consult collector databases for rare parts"
                })
        except Exception:
            # best-effort, ignore on failure
            logger.exception("LEGO detection failed (continuing with base analysis)")

        return analysis

    except Exception as e:
        logger.exception("Failed to analyze artifact image")
        raise RuntimeError(f"Failed to analyze artifact image: {e}") from e


def identify_lego_brick(base64_image: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
    """
    Identify LEGO parts by CLIP image/text similarity using local CLIPModel.
    Returns:
      {
        'is_lego': bool,
        'predictions': [{'label': str, 'similarity': float}, ...],
        'best_similarity': float
      }
    """
    try:
        pil_img = _decode_base64_image_to_pil(base64_image)
        img_emb = _clip_image_embedding(pil_img)

        # build candidate texts (types x colors)
        candidate_texts = []
        for t in _LEGO_TYPES:
            candidate_texts.append(t)
            for c in _LEGO_COLORS:
                candidate_texts.append(f"{c} {t}")
                candidate_texts.append(f"{t} {c}")
        # dedupe while preserving order
        seen = set()
        dedup = []
        for txt in candidate_texts:
            if txt not in seen:
                dedup.append(txt)
                seen.add(txt)
        candidate_texts = dedup

        text_embs = _clip_text_embeddings(candidate_texts)

        sims: List[Tuple[str, float]] = []
        for label, emb in zip(candidate_texts, text_embs):
            try:
                sim = _cosine_similarity(img_emb, emb)
            except Exception:
                sim = -1.0
            sims.append((label, float(sim)))

        # sort and top_k
        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[:top_k]
        predictions = [{'label': lbl, 'similarity': float(sim)} for lbl, sim in top]
        best_similarity = float(predictions[0]['similarity']) if predictions else 0.0
        best_label = predictions[0]['label'] if predictions else ""

        is_lego = ('lego' in best_label.lower() or 'brick' in best_label.lower() or 'minifigure' in best_label.lower()) and best_similarity >= 0.45

        return {
            'is_lego': bool(is_lego),
            'predictions': predictions,
            'best_similarity': best_similarity
        }

    except Exception:
        logger.exception("identify_lego_brick error")
        return None


def compare_with_reference(base64_image: str, reference_artifacts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Compare an input artifact image against reference artifacts (which may include precomputed embeddings).
    Each reference artifact dict expected keys:
      - 'name': str
      - 'embedding': List[float] (optional) OR 'image_base64': str (optional)
      - optional metadata: 'material', 'age', etc.
    Returns:
      {
        'closest_match': str | None,
        'similarity_score': float,
        'differences': [str, ...],
        'alternative_matches': [ {name, similarity, details}, ... ]
      }
    """
    try:
        # compute embedding for input image
        pil_img = _decode_base64_image_to_pil(base64_image)
        query_emb = _clip_image_embedding(pil_img)

        matches = []
        for ref in reference_artifacts:
            sim = calculate_similarity(query_emb, ref)
            matches.append({
                'name': ref.get('name', 'Unknown'),
                'similarity': float(sim),
                'details': {
                    'material': ref.get('material', 'Unknown'),
                    'period': ref.get('age', ref.get('period', 'Unknown'))
                }
            })

        matches.sort(key=lambda x: x['similarity'], reverse=True)
        best = matches[0] if matches else None

        return {
            'closest_match': best['name'] if best else None,
            'similarity_score': best['similarity'] if best else 0.0,
            'differences': generate_differences(best) if best else [],
            'alternative_matches': matches[1:4] if len(matches) > 1 else []
        }
    except Exception:
        logger.exception("compare_with_reference failed")
        return None


def calculate_similarity(embedding1: Any, artifact2: Dict[str, Any]) -> float:
    """
    Calculate cosine similarity between embedding1 (np.ndarray or list) and artifact2.
    artifact2 may contain:
      - 'embedding': list[float] OR
      - 'image_base64': base64 string
    Returns [-1.0, 1.0]
    """
    try:
        # ensure numpy array
        if isinstance(embedding1, (list, tuple)):
            emb1 = np.asarray(embedding1, dtype=float).ravel()
        elif isinstance(embedding1, np.ndarray):
            emb1 = embedding1.ravel()
        else:
            # if it's some wrapper like a tensor, try to convert
            emb1 = np.asarray(embedding1, dtype=float).ravel()

        # normalize emb1
        n1 = np.linalg.norm(emb1)
        if n1 > 0:
            emb1 = emb1 / n1

        # get emb2
        if 'embedding' in artifact2 and artifact2['embedding'] is not None:
            emb2_raw = artifact2['embedding']
            emb2 = np.asarray(emb2_raw, dtype=float).ravel()
        elif 'image_base64' in artifact2 and artifact2['image_base64']:
            pil = _decode_base64_image_to_pil(artifact2['image_base64'])
            emb2 = _clip_image_embedding(pil)
        else:
            return -1.0

        n2 = np.linalg.norm(emb2)
        if n2 > 0:
            emb2 = emb2 / n2

        return _cosine_similarity(emb1, emb2)
    except Exception:
        logger.exception("calculate_similarity error")
        return -1.0


def generate_differences(match: Optional[Dict[str, Any]]) -> List[str]:
    """
    Generate human-readable notes for a match entry (expects 'similarity' and 'details').
    """
    if not match:
        return ["No match available to generate differences."]

    notes: List[str] = []
    sim = match.get('similarity', 0.0)
    details = match.get('details', {})

    # similarity-driven messaging
    if sim >= 0.95:
        notes.append("Highly similar visually (very strong match).")
    elif sim >= 0.8:
        notes.append("Good visual similarity; likely related but verify material/provenance.")
    elif sim >= 0.5:
        notes.append("Some visual similarities but notable differences present.")
    else:
        notes.append("Low visual similarity; likely a different artifact type or style.")

    material = details.get('material')
    period = details.get('period')
    if material and material != "Unknown":
        notes.append(f"Reference material: {material}. Verify physical composition for confirmation.")
    if period and period != "Unknown":
        notes.append(f"Reference period/age: {period}. Cross-check with stratigraphic or provenance data.")

    notes.append("Recommendation: obtain expert physical inspection, provenance records, and high-resolution imaging.")
    return notes


# Example usage (for quick manual test).
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python artifact_analysis_local.py <base64_image_file.txt>")
        print(" where <base64_image_file.txt> contains a single base64-encoded image string.")
        sys.exit(1)
    b64_path = sys.argv[1]
    with open(b64_path, "r") as f:
        b64 = f.read().strip()
    out = analyze_artifact_image(b64)
    import json
    print(json.dumps(out, indent=2))
