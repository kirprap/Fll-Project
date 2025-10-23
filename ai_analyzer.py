"""
AI-powered artifact analysis using Hugging Face Vision Models
"""

import os
import json
import base64
from huggingface_hub import InferenceClient
from pydantic import BaseModel
import numpy as np
from scipy.spatial.distance import cosine


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


def get_huggingface_client():
    """Get Hugging Face client - no API key needed for public models"""
    return InferenceClient()  # No API key needed for inference on public models


def analyze_artifact_image(base64_image):
    """Analyze artifact image using Hugging Face Vision models"""
    try:
        client = get_huggingface_client()
        
        # Convert base64 to bytes
        image_bytes = base64.b64decode(base64_image)
        
        # Use Vision Transformer model for better artifact classification
        result = client.post(
            "google/vit-base-patch16-224",
            data=image_bytes,
            model_kwargs={"return_all_scores": True}
        )
        
        # Get top results
        sorted_results = sorted(result, key=lambda x: x['score'], reverse=True)
        top_result = sorted_results[0]
        
        # Get secondary classifications for better context
        secondary_classes = [r['label'] for r in sorted_results[1:3]]
        
        confidence = top_result['score']
        artifact_type = top_result['label']
        
        # Generate detailed analysis
        analysis = {
            'name': artifact_type,
            'value': "Estimated value requires expert appraisal",
            'age': "Age estimation requires detailed analysis",
            'description': f"This appears to be a {artifact_type}.",
            'cultural_context': "Cultural context requires expert analysis",
            'confidence': confidence,
            'material': "Material analysis requires physical inspection",
            'function': "Function based on artifact type",
            'rarity': "Rarity assessment requires expert evaluation"
        }
        
        return analysis
        
    except Exception as e:
        raise Exception(f"Failed to analyze artifact image: {str(e)}")


def compare_with_reference(base64_image, reference_artifacts):
    """Compare artifacts using visual similarity"""
    try:
        client = get_huggingface_client()
        
        # Convert base64 to bytes
        image_bytes = base64.b64decode(base64_image)
        
        # Use CLIP model for better similarity comparison
        model_id = "sentence-transformers/clip-ViT-B-32"
        
        # Get embeddings for the input image
        image_embedding = client.post(
            f"{model_id}",
            data=image_bytes,
            model_kwargs={"task": "feature-extraction"}
        )
        
        # Compare with reference artifacts
        matches = []
        for ref in reference_artifacts:
            similarity = calculate_similarity(image_embedding, ref)
            matches.append({
                'name': ref['name'],
                'similarity': similarity,
                'details': {
                    'material': ref.get('material', 'Unknown'),
                    'period': ref.get('age', 'Unknown')
                }
            })
        
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'closest_match': matches[0]['name'] if matches else None,
            'similarity_score': matches[0]['similarity'] if matches else 0,
            'differences': generate_differences(matches[0] if matches else None),
            'alternative_matches': matches[1:4] if len(matches) > 1 else []
        }
        
    except Exception as e:
        return None


def calculate_similarity(embedding1, artifact2):
    """Calculate cosine similarity between embeddings"""
    fied for demo
