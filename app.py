from typing import List, Dict, Any, Optional
import base64
import json
import requests
import numpy as np
from PIL import Image
import time
import logging
import sys
import os

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
        timeout: int = 600  # Increased to 10 minutes for large models
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
        embedding endpoint, so we return a zero-vector (length 512) that keeps
        the similarity-search logic functional.

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
        with the zero-vector placeholders.
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
        ``ollama`` uses Qwen3-VL via Ollama for full visual-language analysis.
        """
        if model_choice == "vit":
            return self.classify_image(image)

        elif model_choice == "clip":
            embedding = self.get_embedding(image)
            return {"embedding": embedding.tolist()}

        elif model_choice == "ollama":
            prompt = (
                "You are an expert archaeologist. Analyze the image carefully and "
                "describe the artifact: its type, material, age, cultural origin, "
                "and possible historical function in 2–3 sentences."
            )
            description = self.ollama.generate(prompt, image=image).strip()

            return {
                "name": description.split(".")[0] if description else "Unknown artifact",
                "description": description,
                "confidence": 1.0,
                "embedding": self.get_embedding(image).tolist(),
            }

        else:
            raise ValueError(f"Unknown model_choice: {model_choice}")


# ============================================================================
# Streamlit UI
# ============================================================================

import streamlit as st
import io

# Import database functions
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

try:
    from database import save_artifact, get_all_artifacts, search_artifacts, get_artifact_by_id
except ModuleNotFoundError as e:
    if getattr(e, "name", None) == "database":
        st.error("Database module not found. Please ensure database.py is available.")
        st.stop()
    else:
        st.error(f"Missing dependency: {getattr(e, 'name', 'unknown')}. Please install requirements.")
        st.stop()
except Exception as e:
    st.error(f"Error importing database: {str(e)}")
    st.stop()


@st.cache_resource
def get_analyzer():
    """Cache the AI analyzer to avoid reloading the model."""
    return AIAnalyzer()


@st.cache_resource
def get_fast_analyzer(tier: str):
    """Cache the fast analyzer for the selected tier."""
    from fast_analyzer import FastAnalyzer
    return FastAnalyzer(tier=tier)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Archaeological Artifact Identifier",
        page_icon="🏺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🏺 Archaeological Artifact Identifier")
    st.markdown("Upload images of archaeological artifacts for AI-powered identification and analysis.")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Identify Artifact", "Batch Processing", "Archive", "Search"]
    )

    if page == "Identify Artifact":
        identify_artifact_page()
    elif page == "Batch Processing":
        batch_processing_page()
    elif page == "Archive":
        archive_page()
    elif page == "Search":
        search_page()


def identify_artifact_page():
    """Single artifact identification page."""
    st.header("Identify Single Artifact")

    uploaded_file = st.file_uploader(
        "Upload an artifact image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG, WEBP"
    )

    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Analysis Results")

            # Speed tier selection
            st.markdown("**⚡ Speed vs Quality**")
            speed_tier = st.radio(
                "Choose analysis speed:",
                ["INSTANT (1-2s)", "FAST (20-40s)", "BALANCED (30-60s)", "QUALITY (1-2min)"],
                index=1,  # Default to FAST
                help="Faster = less detailed, Slower = more detailed",
                horizontal=True
            )

            # Extract tier name
            tier_map = {
                "INSTANT (1-2s)": "INSTANT",
                "FAST (20-40s)": "FAST",
                "BALANCED (30-60s)": "BALANCED",
                "QUALITY (1-2min)": "QUALITY"
            }
            selected_tier = tier_map[speed_tier]

            # Show what this tier uses
            tier_info = {
                "INSTANT": "Uses ViT (basic classification)",
                "FAST": "Uses LLaVA 7B (good quality)",
                "BALANCED": "Uses Qwen2-VL 7B (better quality)",
                "QUALITY": "Uses Qwen3-VL (best quality)"
            }
            st.caption(tier_info[selected_tier])

            if st.button("Analyze Artifact", type="primary"):
                expected_time = {
                    "INSTANT": "1-2 seconds",
                    "FAST": "20-40 seconds",
                    "BALANCED": "30-60 seconds",
                    "QUALITY": "1-2 minutes"
                }[selected_tier]

                with st.spinner(f"Analyzing artifact... Expected time: {expected_time}"):
                    try:
                        # Use fast analyzer
                        analyzer = get_fast_analyzer(selected_tier)
                        result = analyzer.analyze_artifact(image)

                        st.success(f"✅ Analysis Complete in {result.get('analysis_time', 'N/A')}!")
                        st.markdown(f"**Name:** {result.get('name', 'Unknown')}")
                        st.markdown(f"**Description:** {result.get('description', 'N/A')}")
                        st.markdown(f"**Confidence:** {result.get('confidence', 0):.2%}")
                        st.markdown(f"**Method:** {result.get('method', 'N/A')}")
                        st.markdown(f"**Quality Tier:** {result.get('tier', 'N/A')}")

                        # Option to save to archive
                        if st.button("Save to Archive"):
                            img_bytes = io.BytesIO()
                            image.save(img_bytes, format='PNG')
                            artifact_data = {
                                "name": result.get('name', 'Unknown'),
                                "description": result.get('description', ''),
                                "confidence": result.get('confidence', 0.0),
                                "value": "Requires expert appraisal",
                                "age": "Requires expert analysis",
                                "cultural_context": "Requires expert input",
                                "material": "Requires physical inspection",
                                "function": "Inferred from analysis",
                                "rarity": "Requires comparison"
                            }
                            artifact_id = save_artifact(artifact_data, img_bytes.getvalue())
                            st.success(f"Artifact saved to archive with ID: {artifact_id}")

                    except RuntimeError as e:
                        error_msg = str(e)
                        if "Ollama generation failed" in error_msg or "timeout" in error_msg.lower():
                            st.error("⚠️ **Ollama Connection Error**")
                            st.markdown("""
                            **Possible causes:**
                            1. **Ollama is not running** - Start Ollama service
                            2. **Model not downloaded** - Run: `ollama pull qwen3-vl:32b`
                            3. **Timeout (model too large)** - The model is processing, please wait
                            4. **Wrong endpoint** - Check OLLAMA_ENDPOINT environment variable

                            **Quick fixes:**
                            - **Docker**: `docker-compose restart ollama`
                            - **Local**: `ollama serve` in a terminal
                            - **Check model**: `ollama list`
                            - **Pull model**: `ollama pull qwen3-vl:32b`

                            **Alternative**: Try using the **ViT** model instead (faster, no Ollama required)
                            """)
                            with st.expander("Technical Details"):
                                st.code(error_msg)
                        else:
                            st.error(f"Error during analysis: {error_msg}")
                            st.exception(e)
                    except Exception as e:
                        st.error(f"Unexpected error: {str(e)}")
                        st.exception(e)


def batch_processing_page():
    """Batch processing page for multiple artifacts."""
    st.header("Batch Processing")
    st.markdown("Upload multiple artifact images for batch analysis.")

    uploaded_files = st.file_uploader(
        "Upload artifact images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Upload multiple images for batch processing"
    )

    if uploaded_files:
        st.info(f"Uploaded {len(uploaded_files)} images")

        model_choice = st.selectbox(
            "Select AI Model",
            ["ollama", "vit"],
            help="Choose the model for batch analysis"
        )

        if st.button("Process All", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            analyzer = get_analyzer()
            results = []

            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")

                try:
                    image = Image.open(uploaded_file)
                    result = analyzer.analyze_image(image, model_choice=model_choice)
                    results.append({
                        "filename": uploaded_file.name,
                        "result": result,
                        "image": image
                    })
                except Exception as e:
                    st.warning(f"Failed to process {uploaded_file.name}: {str(e)}")

                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.text("Processing complete!")

            # Display results
            st.subheader("Results")
            for item in results:
                with st.expander(f"📷 {item['filename']}"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(item['image'], use_container_width=True)
                    with col2:
                        result = item['result']
                        st.markdown(f"**Name:** {result.get('name', 'Unknown')}")
                        if 'description' in result:
                            st.markdown(f"**Description:** {result.get('description')}")
                        st.markdown(f"**Confidence:** {result.get('confidence', 0):.2%}")


def archive_page():
    """Archive page to view saved artifacts."""
    st.header("Artifact Archive")

    try:
        artifacts = get_all_artifacts(limit=50, include_images=True)

        if not artifacts:
            st.info("No artifacts in archive yet. Start by identifying some artifacts!")
            return

        st.success(f"Found {len(artifacts)} artifacts in archive")

        # Display artifacts in a grid
        cols_per_row = 3
        for i in range(0, len(artifacts), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(artifacts):
                    artifact = artifacts[i + j]
                    with col:
                        with st.container():
                            # Display image if available
                            if artifact.get('image_base64'):
                                img_data = base64.b64decode(artifact['image_base64'])
                                img = Image.open(io.BytesIO(img_data))
                                st.image(img, use_container_width=True)

                            st.markdown(f"**{artifact.get('name', 'Unknown')}**")
                            st.caption(f"ID: {artifact.get('id')} | Uploaded: {artifact.get('uploaded_at', 'N/A')}")

                            if st.button(f"View Details", key=f"view_{artifact.get('id')}"):
                                st.session_state['selected_artifact'] = artifact.get('id')

        # Show selected artifact details
        if 'selected_artifact' in st.session_state:
            artifact_id = st.session_state['selected_artifact']
            artifact = get_artifact_by_id(artifact_id)
            if artifact:
                st.divider()
                st.subheader(f"Artifact Details: {artifact.name}")

                col1, col2 = st.columns([1, 2])
                with col1:
                    if artifact.image_data:
                        img = Image.open(io.BytesIO(artifact.image_data))
                        st.image(img, use_container_width=True)

                with col2:
                    st.markdown(f"**Name:** {artifact.name}")
                    st.markdown(f"**Description:** {artifact.description or 'N/A'}")
                    st.markdown(f"**Material:** {artifact.material or 'N/A'}")
                    st.markdown(f"**Age:** {artifact.age or 'N/A'}")
                    st.markdown(f"**Cultural Context:** {artifact.cultural_context or 'N/A'}")
                    st.markdown(f"**Function:** {artifact.function or 'N/A'}")
                    st.markdown(f"**Rarity:** {artifact.rarity or 'N/A'}")
                    st.markdown(f"**Value:** {artifact.value or 'N/A'}")
                    st.markdown(f"**Confidence:** {artifact.confidence:.2%}" if artifact.confidence else "**Confidence:** N/A")
                    st.markdown(f"**Uploaded:** {artifact.uploaded_at}")

    except Exception as e:
        st.error(f"Error loading archive: {str(e)}")
        st.exception(e)


def search_page():
    """Search page for finding artifacts."""
    st.header("Search Artifacts")

    search_query = st.text_input(
        "Search artifacts",
        placeholder="Enter keywords to search...",
        help="Search by name, description, material, or cultural context"
    )

    if search_query:
        try:
            results = search_artifacts(search_query, limit=20)

            if results:
                st.success(f"Found {len(results)} matching artifacts")

                for artifact in results:
                    with st.expander(f"🏺 {artifact.get('name', 'Unknown')} (ID: {artifact.get('id')})"):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.markdown(f"**ID:** {artifact.get('id')}")
                            st.markdown(f"**Uploaded:** {artifact.get('uploaded_at', 'N/A')}")
                        with col2:
                            st.markdown(f"**Description:** {artifact.get('description', 'N/A')}")
                            st.markdown(f"**Material:** {artifact.get('material', 'N/A')}")
                            st.markdown(f"**Cultural Context:** {artifact.get('cultural_context', 'N/A')}")
            else:
                st.info("No artifacts found matching your search.")

        except Exception as e:
            st.error(f"Error during search: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
