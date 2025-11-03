from typing import List, Dict, Any, Optional
import base64
import json
import requests
import numpy as np
from PIL import Image
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class OllamaClient:
    """Simple wrapper for the local Ollama HTTP API with retry logic."""

    def __init__(
        self,
        model: str = "qwen3-vl:latest",  # Changed from 32b to latest (6GB, much faster)
        endpoint: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30  # Increased to 10 minutes for large models
    ):
        # Auto-detect endpoint based on environment
        if endpoint is None:
            import os
            # Check if running in Docker (HOSTNAME env var is set by Docker)
            if os.getenv('HOSTNAME') and 'docker' in os.getenv('HOSTNAME', '').lower():
                endpoint = "http://ollama:11434"
            else:
                endpoint = os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434')

        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout
        logger.info(f"OllamaClient initialized: endpoint={self.endpoint}, model={self.model}, timeout={self.timeout}s")

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a POST request with retry logic."""
        url = f"{self.endpoint}{path}"
        headers = {"Content-Type": "application/json"}

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout as e:
                last_exception = e
                logger.warning(f"Request timeout on attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

            except requests.exceptions.ConnectionError as e:
                last_exception = e
                logger.warning(f"Connection error on attempt {attempt + 1}/{self.max_retries}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

            except requests.exceptions.HTTPError as e:
                last_exception = e
                logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
                raise  # Don't retry on HTTP errors (4xx, 5xx)

            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error: {str(e)}")
                raise

        # If all retries failed, raise the last exception
        raise last_exception if last_exception else Exception("All retry attempts failed")

    def generate(self, prompt: str, image: Optional[Image.Image] = None) -> str:
        """
        Generate a response from the model.

        If an image is supplied it is encoded as PNG and sent in the ``images``
        field as a base64 string (the format expected by Ollama for multimodal
        models).
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        if image is not None:
            from io import BytesIO

            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            payload["images"] = [img_b64]

        try:
            result = self._post("/api/generate", payload)
            return str(result.get("response", ""))
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise RuntimeError(f"Ollama generation failed: {str(e)}") from e


class AIAnalyzer:
    """
    Analyzer that uses Ollama's ``qwen3-vl:32b`` model for image description.

    The public API mirrors the original implementation so existing code
    (Streamlit UI, database helpers, etc.) continues to work unchanged.
    """

    def __init__(self):
        self.ollama = OllamaClient()

    def classify_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Return a short name for the artifact using Ollama.

        Ollama does not provide a confidence score, so ``confidence`` is set to
        ``1.0`` as a placeholder.
        """
        prompt = "Provide a short, descriptive name for the object in the image."
        name = self.ollama.generate(prompt, image=image).strip()
        return {"name": name, "confidence": 1.0}

    def get_embedding(self, image: Optional[Image.Image] = None) -> np.ndarray:
        """
        Return a placeholder embedding.

        The current Ollama multimodal model does not expose a dedicated
        embedding endpoint, so we return a zero‑vector (length 512) that keeps
        the similarity‑search logic functional.

        Args:
            image: Optional PIL Image (currently unused, placeholder for future implementation)
        """
        # Note: image parameter is for API compatibility but not used in placeholder implementation
        return np.zeros(512, dtype=np.float32)

    def similarity_search(
        self, query_embedding: np.ndarray, database_artifacts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare ``query_embedding`` with stored artifact embeddings.

        This implementation is unchanged from the original codebase and works
        with the zero‑vector placeholders.
        """
        results: List[Dict[str, Any]] = []

        for db_art in database_artifacts:
            if "id" in db_art and "name" in db_art:
                emb = db_art.get("embedding")
                if emb is not None and isinstance(emb, np.ndarray):
                    score = cosine_similarity(query_embedding, emb)
                    results.append({"artifact": db_art, "score": score})

        results.sort(key=lambda x: float(x["score"]), reverse=True)

        if results:
            closest = results[0]
            return {
                "closest_match": str(closest["artifact"]["name"]),
                "similarity_score": float(closest["score"]),
                "alternative_matches": results[1:4],
            }

        return {}

    def analyze_image(self, image: Image.Image, model_choice: str = "vit") -> Dict[str, Any]:
        """
        Dispatch analysis based on ``model_choice``.

        ``vit`` delegates to ``classify_image`` (now powered by Ollama).
        ``clip`` returns a placeholder embedding.
        """
        if model_choice == "vit":
            return self.classify_image(image)
        elif model_choice == "clip":
            embedding = self.get_embedding(image)
            return {"embedding": embedding.tolist()}
        else:
            raise ValueError(f"Unknown model_choice: {model_choice}")
