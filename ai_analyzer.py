"""
AI-powered artifact analysis using Hugging Face Vision Models
"""

import os
import base64
import logging
import traceback
from typing import List, Dict, Optional, Any, Tuple
from typing import TYPE_CHECKING
import io

# Provide typing-only imports for editors / static checkers and safe runtime fallbacks
if TYPE_CHECKING:
    # These imports are only needed for static type checking (Pylance, mypy)
    from huggingface_hub import InferenceClient  # type: ignore
    from pydantic import BaseModel  # type: ignore
    import numpy as np  # type: ignore
    from scipy.spatial.distance import cosine  # type: ignore
else:
    # Runtime imports: attempt to import, but provide safe fallbacks so the
    # file can be opened and analyzed in editors that don't have these
    # dependencies installed in the workspace.
    try:
        from huggingface_hub import InferenceClient  # type: ignore[reportMissingImports]
    except Exception:  # pragma: no cover - editor/runtime fallback
        InferenceClient = None  # type: ignore

    try:
        from pydantic import BaseModel  # type: ignore[reportMissingImports]
    except Exception:  # pragma: no cover - fallback for editors without pydantic
        class BaseModel:  # type: ignore
            pass

    try:
        import numpy as np  # type: ignore[reportMissingImports]
    except Exception:  # pragma: no cover - numpy missing
        np = None  # type: ignore

    try:
        from scipy.spatial.distance import cosine  # type: ignore[reportMissingImports]
    except Exception:  # pragma: no cover - scipy missing
        def cosine(a, b):
            raise RuntimeError("scipy.spatial.distance.cosine is not available")

    # Optional local inference helpers (used as a fallback when remote API fails).
    try:
        from transformers import pipeline as _hf_pipeline  # type: ignore[reportMissingImports]
    except Exception:  # pragma: no cover - transformers not installed
        _hf_pipeline = None  # type: ignore

    try:
        from PIL import Image  # type: ignore[reportMissingImports]
    except Exception:  # pragma: no cover - Pillow not installed
        Image = None  # type: ignore

    # Cache for a locally created pipeline to avoid re-loading model repeatedly
    _LOCAL_PIPELINE: Optional[Any] = None

    def _get_local_image_pipeline() -> Any:
        """Create (and cache) a local transformers image-classification pipeline.

        Raises RuntimeError if `transformers` isn't available.
        """
        nonlocal _LOCAL_PIPELINE
        if _hf_pipeline is None:
            raise RuntimeError("transformers.pipeline is not available in this environment")
        if _LOCAL_PIPELINE is None:
            # create pipeline with the same model id used for remote inference
            _LOCAL_PIPELINE = _hf_pipeline("image-classification", model=_VIT_MODEL)
        return _LOCAL_PIPELINE

# Ensure optional local-fallback symbols exist at module level for static analysis
try:
    _hf_pipeline  # type: ignore
except NameError:
    _hf_pipeline = None  # type: ignore

try:
    Image  # type: ignore
except NameError:
    Image = None  # type: ignore

try:
    _get_local_image_pipeline  # type: ignore
except NameError:
    def _get_local_image_pipeline() -> Any:  # type: ignore
        raise RuntimeError("transformers.pipeline is not available in this environment")


# Load optional token from config (keeps token management centralized)
# Annotate to avoid constant redefinition warnings from linters
huggingface_token: Optional[str] = None
try:
    # Prefer a config-provided token if available
    from config import HUGGINGFACE_TOKEN as _CFG_HF_TOKEN  # type: ignore
    huggingface_token = _CFG_HF_TOKEN
except Exception:
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

class ArtifactAnalysis(BaseModel):
    name: str
    value: str
    age: str
    description: str
    cultural_context: str
    confidence: float
    material: str
    function: str
    rarity: str


# Default model IDs
# Use a public image-classification model to avoid gated-model access issues during testing.
# Switched to ResNet-50 (public) as a robust image-classification baseline for inference API.
_VIT_MODEL = "microsoft/resnet-50"
_CLIP_MODEL = "sentence-transformers/clip-ViT-B-32"

# Lego-specific heuristics
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


def get_huggingface_client() -> Any:
    """Get Hugging Face client. If a token is provided in environment/config it will be used."""
    token = huggingface_token if 'huggingface_token' in globals() else None
    if not token:
        # fallback to known env names
        token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if token:
        return InferenceClient(token=token)
    # unauthenticated client (public access) � may be rate-limited
    return InferenceClient()


def _call_inference(client: Any, model: str, data: Any = None, model_kwargs: Optional[Dict[str, Any]] = None, **extra) -> Any:
    """Compatibility wrapper for InferenceClient.post across versions.

    Some versions expect `model_kwargs=...`, others expect `params=...` or
    `inputs`/`data` positional args. Try the common call first and fall back
    to alternatives when a TypeError about unexpected kwargs is raised.
    """
    if client is None:
        raise RuntimeError("Inference client is not available")

    # Basic validation
    if model is None:
        raise ValueError("No model specified for inference call")

    # Prefer calling with explicit 'task' when provided in model_kwargs (some hf versions accept a `task` kw)
    if model_kwargs and isinstance(model_kwargs, dict) and 'task' in model_kwargs:
        task_val = model_kwargs.get('task')
        return client.post(model=model, data=data, task=task_val, **extra)

    # Many HF client versions accept the request body as 'data' (bytes) and
    # additional model parameters inside the JSON payload under 'parameters'.
    # Different model endpoints are picky about whether binary 'data' is sent
    # with a separate JSON body. Empirically the safest approach is:
    # - If data is raw bytes (image) -> try data-only first
    # - If data is a list of strings (text batch) -> send as json inputs
    # - Otherwise try JSON parameters variants
    try_variants = []

    # If binary image bytes, prefer data-only first to avoid multipart/content-type issues
    if isinstance(data, (bytes, bytearray)):
        try_variants.append({'data': data})
        if model_kwargs is not None:
            try_variants.append({'data': data, 'json': {'parameters': model_kwargs}})
            try_variants.append({'data': data, 'json': model_kwargs})
    # If data looks like a list of textual inputs, send as json 'inputs'
    elif isinstance(data, (list, tuple)) and all(isinstance(x, str) for x in data):
        if model_kwargs is not None:
            try_variants.append({'json': {'inputs': data, 'parameters': model_kwargs}})
        try_variants.append({'json': {'inputs': data}})
    else:
        # Fallback: try sending model_kwargs in json then plain data
        if model_kwargs is not None:
            try_variants.append({'data': data, 'json': {'parameters': model_kwargs}})
            try_variants.append({'data': data, 'json': model_kwargs})
        try_variants.append({'data': data})

    last_exc: Optional[BaseException] = None
    for kwargs in try_variants:
        try:
            return client.post(model=model, **kwargs)
        except Exception as e:
            # Capture context to help debugging (caller will see wrapped message)
                last_exc = e
                continue

            # If none of the call styles worked, raise a detailed error with context
    if last_exc:
        msg = (
            f"Inference call failed for model={model!r} with client={type(client).__name__}."
            f" Tried variants: {try_variants!r}. Last error: {last_exc!r}"
        )
        raise RuntimeError(msg) from last_exc
    # Fallback generic call (shouldn't be reached)
    try:
        return client.post(model, data=data, **extra)
    except Exception as e:
        raise RuntimeError(f"Final fallback inference call failed: {e}") from e


def _response_to_embedding(resp: Any) -> Any:
    """
    Normalize various inference responses into a 1-D NumPy array.
    The InferenceClient may return lists, nested lists, or already-converted arrays.
    This returns a 1-D array for a single embedding input.
    """
    if resp is None:
        raise ValueError("Empty response for embedding conversion")

    if isinstance(resp, np.ndarray):
        return resp.ravel()

    if isinstance(resp, dict):
        # common wrapper keys
        for key in ("embedding", "features", "vector"):
            if key in resp:
                return _response_to_embedding(resp[key])

    if isinstance(resp, (list, tuple)):
        # If it's a nested list representing a single embedding, convert to 1D
        try:
            arr = np.array(resp, dtype=float)
            # If shape is (1, N) or (N,) -> ravel to 1D
            return arr.ravel()
        except Exception as e:
            raise TypeError(f"Unable to convert response to embedding: {e}")

    raise TypeError(f"Unsupported embedding response type: {type(resp)}")


def _batch_response_to_embeddings(resp: Any) -> List[Any]:
    """
    Convert a batched response (list of embedding responses) into a list of 1-D arrays.
    """
    if resp is None:
        return []
    if isinstance(resp, (list, tuple)):
        embeddings = []
        for item in resp:
            embeddings.append(_response_to_embedding(item))
        return embeddings
    # Single embedding provided where a batch was expected
    return [_response_to_embedding(resp)]


def analyze_artifact_image(base64_image: str) -> Dict[str, Any]:
    """Analyze artifact image using a vision classifier and return structured analysis."""
    try:
        client = get_huggingface_client()

        # Validate input
        if not base64_image or not isinstance(base64_image, str):
            raise ValueError("No image data provided or input is not a base64 string")

        # Convert base64 to bytes (guard against invalid base64)
        try:
            image_bytes = base64.b64decode(base64_image)
        except Exception as e:
            raise ValueError(f"Invalid base64 image data: {e}") from e

        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Decoded image bytes are empty")

        # Request classification scores from ViT (returns list of {label,score})
        try:
            result = _call_inference(client, _VIT_MODEL, data=image_bytes, model_kwargs={"return_all_scores": True})
        except Exception as e:
            # Attach a helpful message for troubleshooting (401/403, model not found, or API errors)
            logging.getLogger(__name__).exception("ViT classification failed")
            # Try local transformers pipeline fallback if available to avoid gated remote models
            if _hf_pipeline is not None and Image is not None:
                try:
                    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    local_pipe = _get_local_image_pipeline()
                    preds = local_pipe(pil_img)
                    # `preds` from transformers pipeline is a list of {label, score}
                    result = preds
                except Exception:
                    logging.getLogger(__name__).exception("Local transformers pipeline failed")
                    raise Exception(f"Inference error while classifying image: {e}") from e
            else:
                raise Exception(f"Inference error while classifying image: {e}") from e

        # Validate result format
        if not isinstance(result, (list, tuple)):
            raise ValueError("Unexpected classification response format")

        # Sort and pick top results
        sorted_results = sorted(result, key=lambda x: x.get('score', 0.0), reverse=True)
        top_result = sorted_results[0] if sorted_results else {"label": "unknown", "score": 0.0}

        # Secondary classes for context
        secondary_classes = [r.get('label', '') for r in sorted_results[1:3]]

        confidence = float(top_result.get('score', 0.0))
        artifact_type = str(top_result.get('label', 'Unknown'))

        # Prepare base analysis
        analysis = {
            'name': artifact_type,
            'value': "Estimated value requires expert appraisal",
            'age': "Age estimation requires detailed material/stratigraphic analysis",
            'description': f"This image is classified as \"{artifact_type}\". Secondary possibilities: {secondary_classes}.",
            'cultural_context': "Cultural context requires expert analysis and provenance data",
            'confidence': confidence,
            'material': "Material analysis requires physical or spectral inspection",
            'function': "Function inferred from type; needs expert confirmation",
            'rarity': "Rarity assessment requires provenance and collection comparison"
        }

        # Attempt Lego recognition if classifier suggests a brick or if explicit user interest
        try:
            lego_info = identify_lego_brick(base64_image, client=client, top_k=5)
            if lego_info and lego_info.get('is_lego') and lego_info.get('best_similarity', 0.0) >= _LEGO_CONFIDENCE_THRESHOLD:
                best_label = lego_info['predictions'][0]['label']
                best_sim = float(lego_info['predictions'][0]['similarity'])
                analysis.update({
                    'name': f"LEGO: {best_label}",
                    'description': f"Detected LEGO component: {best_label} (confidence {best_sim:.2f}). Secondary: {[p['label'] for p in lego_info['predictions'][1:3]]}",
                    'confidence': max(confidence, best_sim),
                    'material': "ABS plastic (typical for LEGO bricks)",
                    'function': "Toy / construction element",
                    'rarity': "Common (verify special collector variants separately)",
                    'value': "Typical retail/collector value; consult collector databases for rare parts"
                })
        except Exception:
            # Lego detection is best-effort; ignore failures and return base analysis
            pass

        return analysis

    except Exception as e:
        raise Exception(f"Failed to analyze artifact image: {str(e)}")


def identify_lego_brick(base64_image: str, client: Optional[Any] = None, top_k: int = 5) -> Optional[Dict[str, Any]]:
    """
    Attempt to identify LEGO bricks/parts from an image using CLIP-style image-text similarity.
    Returns:
      - is_lego: bool
      - predictions: list of top_k {label, similarity}
      - best_similarity: float
    This is heuristic-based and should be treated as a best-effort classifier.
    """
    try:
        if client is None:
            client = get_huggingface_client()

        image_bytes = base64.b64decode(base64_image)
        image_embedding_resp = _call_inference(client, _CLIP_MODEL, data=image_bytes, model_kwargs={"task": "feature-extraction"})
        image_embedding = _response_to_embedding(image_embedding_resp)

        # Build candidate textual prompts (types x colors)
        candidate_texts = []
        for t in _LEGO_TYPES:
            candidate_texts.append(t)
            for c in _LEGO_COLORS:
                candidate_texts.append(f"{c} {t}")
                candidate_texts.append(f"{t} {c}")
        # Deduplicate
        candidate_texts = list(dict.fromkeys(candidate_texts))

        # Get embeddings for candidate texts in batch
        text_embeddings_resp = _call_inference(client, _CLIP_MODEL, data=candidate_texts, model_kwargs={"task": "feature-extraction"})
        text_embeddings = _batch_response_to_embeddings(text_embeddings_resp)

        # Compute similarities
        sims: List[Tuple[str, float]] = []
        for label, emb in zip(candidate_texts, text_embeddings):
            try:
                if np.allclose(emb, 0) or np.allclose(image_embedding, 0):
                    sim = -1.0
                else:
                    dist = cosine(image_embedding, emb)
                    sim = 1.0 - dist
                    sim = float(max(-1.0, min(1.0, sim)))
            except Exception:
                sim = -1.0
            sims.append((label, sim))

        # Sort and pick top_k
        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[:top_k]

        predictions = [{'label': lbl, 'similarity': float(sim)} for lbl, sim in top]
        best_similarity = predictions[0]['similarity'] if predictions else 0.0

        # Heuristic: if top label contains 'LEGO' or 'brick' and similarity is reasonable, treat as lego
        best_label = predictions[0]['label'] if predictions else ""
        is_lego = ('lego' in best_label.lower() or 'brick' in best_label.lower() or 'minifigure' in best_label.lower()) and best_similarity >= 0.45

        return {
            'is_lego': bool(is_lego),
            'predictions': predictions,
            'best_similarity': float(best_similarity)
        }

    except Exception:
        return None


def compare_with_reference(base64_image: str, reference_artifacts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Compare an input artifact image against a list of reference artifacts.
    Each reference artifact is expected to be a dict containing at least:
      - 'name': str
      - either 'embedding': List[float] or 'image_base64': str
      - optional metadata like 'material' or 'age'

    Returns a summary with the closest match and alternatives.
    """
    try:
        client = get_huggingface_client()

        # Convert base64 to bytes and get embedding for input image
        image_bytes = base64.b64decode(base64_image)
        image_embedding_resp = _call_inference(client, _CLIP_MODEL, data=image_bytes, model_kwargs={"task": "feature-extraction"})
        image_embedding = _response_to_embedding(image_embedding_resp)

        matches = []
        for ref in reference_artifacts:
            sim = calculate_similarity(image_embedding, ref, client=client)
            matches.append({
                'name': ref.get('name', 'Unknown'),
                'similarity': sim,
                'details': {
                    'material': ref.get('material', 'Unknown'),
                    'period': ref.get('age', ref.get('period', 'Unknown'))
                }
            })

        # Sort by similarity descending
        matches.sort(key=lambda x: x['similarity'], reverse=True)

        best = matches[0] if matches else None

        return {
            'closest_match': best['name'] if best else None,
            'similarity_score': best['similarity'] if best else 0.0,
            'differences': generate_differences(best) if best else [],
            'alternative_matches': matches[1:4] if len(matches) > 1 else []
        }

    except Exception:
        # Keep caller logic simple; caller may log or raise further
        return None


def calculate_similarity(embedding1: Any, artifact2: Dict[str, Any], client: Optional[Any] = None) -> float:
    """
    Calculate cosine similarity between a provided embedding and a reference artifact.
    - embedding1: embedding vector or inference response for the query image
    - artifact2: reference artifact dict. Prefer 'embedding' key (list/array). If missing,
                 'image_base64' will be used to compute an embedding (requires client).
    - client: optional InferenceClient instance used to compute embeddings for artifact2 if needed.

    Returns a similarity score in the range [-1, 1] where 1 is identical.
    """
    try:
        # Normalize embedding1
        emb1 = _response_to_embedding(embedding1)

        # Get embedding for artifact2
        if 'embedding' in artifact2:
            emb2 = _response_to_embedding(artifact2['embedding'])
        elif 'image_base64' in artifact2:
            if client is None:
                client = get_huggingface_client()
            image_bytes = base64.b64decode(artifact2['image_base64'])
            resp = _call_inference(client, _CLIP_MODEL, data=image_bytes, model_kwargs={"task": "feature-extraction"})
            emb2 = _response_to_embedding(resp)
        else:
            # No embedding or image to compare to; return minimal similarity
            return -1.0

        # If either embedding is all zeros, avoid division by zero
        if np.allclose(emb1, 0) or np.allclose(emb2, 0):
            return -1.0

        # Cosine distance -> similarity
        dist = cosine(emb1, emb2)
        # scipy.spatial.distance.cosine returns 1 - cosine_similarity, so:
        similarity = 1.0 - dist

        # Clamp numeric errors
        similarity = float(max(-1.0, min(1.0, similarity)))
        return similarity

    except Exception:
        return -1.0


def generate_differences(match: Optional[Dict[str, Any]]) -> List[str]:
    """
    Generate a human-readable list of differences / notes for the provided match entry.
    If match is None or similarity is low, returns conservative notes.
    """
    if not match:
        return ["No match available to generate differences."]

    notes: List[str] = []
    sim = match.get('similarity', 0.0)
    details = match.get('details', {})

    # Similarity-based notes
    if sim >= 0.95:
        notes.append("Highly similar visually (very strong match).")
    elif sim >= 0.8:
        notes.append("Good visual similarity; likely related but verify material/provenance.")
    elif sim >= 0.5:
        notes.append("Some visual similarities but notable differences present.")
    else:
        notes.append("Low visual similarity; likely a different artifact type or style.")

    # Add metadata observations when available
    material = details.get('material')
    period = details.get('period')
    if material and material != "Unknown":
        notes.append(f"Reference material: {material}. Verify physical composition for confirmation.")
    if period and period != "Unknown":
        notes.append(f"Reference period/age: {period}. Cross-check with stratigraphic or provenance data.")

    # Guidance for next steps
    notes.append("Recommendation: obtain expert physical inspection, provenance records, and high-resolution imaging.")

    return notes
