from typing import List, Dict
import torch
import numpy as np
from PIL import Image
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    CLIPModel,
    CLIPProcessor,
)


# ------------------------------
# Cosine similarity
# ------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ------------------------------
# Model Manager
# ------------------------------
class ModelManager:
    def __init__(self):
        self.vit_model = None
        self.vit_processor = None
        self.clip_model = None
        self.clip_processor = None

    def load_vit(self, model_name="google/vit-base-patch16-224"):
        if self.vit_model is None:
            self.vit_model = ViTForImageClassification.from_pretrained(model_name)
            self.vit_processor = ViTImageProcessor.from_pretrained(model_name)

    def load_clip(self, model_name="openai/clip-vit-base-patch32"):
        if self.clip_model is None:
            self.clip_model = CLIPModel.from_pretrained(model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)


# ------------------------------
# Analyzer
# ------------------------------
class AIAnalyzer:
    def __init__(self):
        self.models = ModelManager()

    def classify_image(self, image: Image.Image) -> Dict[str, float]:
        self.models.load_vit()
        inputs = self.models.vit_processor(images=image, return_tensors="pt")
        outputs = self.models.vit_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        top_idx = probs.argmax().item()
        top_score = probs[0, top_idx].item()
        top_label = self.models.vit_model.config.id2label[top_idx]
        return {"name": top_label, "confidence": top_score}

    def get_embedding(self, image: Image.Image) -> np.ndarray:
        self.models.load_clip()
        inputs = self.models.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.models.clip_model.get_image_features(**inputs)
        return embeddings[0].cpu().numpy()

    def similarity_search(
        self, query_embedding: np.ndarray, database_artifacts: List[Dict]
    ) -> Dict:
        """Compare query embedding with database artifacts."""
        results = []

        for db_art in database_artifacts:
            # Get the image data and recreate embedding

            # In a production app, you would store embeddings in the database
            # For now, we'll skip artifacts that don't have the data we need
            if "id" in db_art and "name" in db_art:
                # You would normally retrieve stored embeddings from the database
                # For this example, we'll create a placeholder
                # In a real implementation, you'd store the embedding when saving the artifact
                emb = db_art.get("embedding")

                if emb is not None:
                    score = cosine_similarity(query_embedding, emb)

                    results.append({"artifact": db_art, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)

        if results:
            closest = results[0]

            return {
                "closest_match": closest["artifact"]["name"],
                "similarity_score": closest["score"],
                "alternative_matches": results[1:4],
            }

        return {}

    def analyze_image(self, image: Image.Image, model_choice="vit") -> Dict:
        if model_choice == "vit":
            return self.classify_image(image)
        elif model_choice == "clip":
            embedding = self.get_embedding(image)
            return {"embedding": embedding}
        else:
            raise ValueError(f"Unknown model_choice: {model_choice}")
