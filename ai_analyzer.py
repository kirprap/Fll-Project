"""
AI-powered artifact analysis using Google Gemini Vision API
"""

import os
import json
import base64
from google import genai
from google.genai import types
from pydantic import BaseModel

# IMPORTANT: KEEP THIS COMMENT
# Follow these instructions when using this blueprint:
# - Note that the newest Gemini model series is "gemini-2.5-flash" or gemini-2.5-pro"
#   - do not change this unless explicitly requested by the user
# - Sometimes the google genai SDK has occasional type errors. You might need to run to validate, at time.  
# The SDK was recently renamed from google-generativeai to google-genai. This file reflects the new name and the new APIs.

# This API key is from Gemini Developer API Key, not vertex AI API Key
def get_gemini_client():
    """Get Gemini client, raising a user-friendly error if API key is missing"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured. Please add your Gemini API key in the Secrets panel.")
    return genai.Client(api_key=api_key)


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


def analyze_artifact_image(base64_image):
    """
    Analyze an artifact image using Google Gemini Vision API
    Returns structured data about the artifact including name, value, age, and description
    """
    
    try:
        # Get Gemini client
        client = get_gemini_client()
        
        # Convert base64 to bytes
        image_bytes = base64.b64decode(base64_image)
        
        prompt = """
        You are an expert archaeologist and artifact specialist. Analyze this image of an archaeological artifact and provide detailed information.

        Please respond with a JSON object containing the following fields:
        - name: The specific name/type of the artifact
        - value: Estimated monetary value in USD (number only, no currency symbol)
        - age: Historical period or approximate age (e.g., "Roman Period, 1st-2nd Century AD" or "Paleolithic, 30,000 years ago")
        - description: Detailed description of the artifact, its characteristics, and significance
        - cultural_context: Information about the culture or civilization that created it
        - confidence: Your confidence level in this identification (0.0 to 1.0)
        - material: Primary material(s) the artifact is made from
        - function: The likely purpose or use of this artifact
        - rarity: How common or rare this type of artifact is (common/uncommon/rare/very rare)

        Base your analysis on:
        1. Visual characteristics (shape, size, decoration, wear patterns)
        2. Material appearance
        3. Manufacturing techniques visible
        4. Style and artistic elements
        5. Functional design features

        If you cannot identify the artifact with confidence, indicate this in your response and provide the best possible analysis based on what you can observe.
        """

        system_instruction = "You are an expert archaeological analyst. Provide detailed, accurate analysis of artifacts based on visual evidence. Always respond in valid JSON format."

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
                prompt
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=ArtifactAnalysis,
            ),
        )
        
        # Parse the JSON response
        if response.text:
            result = json.loads(response.text)
            
            # Validate and clean the result
            validated_result = validate_analysis_result(result)
            
            return validated_result
        else:
            raise Exception("Empty response from Gemini API")
        
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse AI response as JSON: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to analyze artifact image: {str(e)}")


def validate_analysis_result(result):
    """
    Validate and clean the analysis result from the AI
    Ensures all required fields are present and properly formatted
    """
    
    validated = {
        'name': result.get('name', 'Unknown Artifact'),
        'value': str(result.get('value', 'Unknown')),
        'age': result.get('age', 'Unknown Period'),
        'description': result.get('description', 'No description available'),
        'cultural_context': result.get('cultural_context', 'Unknown cultural context'),
        'confidence': float(result.get('confidence', 0.5)),
        'material': result.get('material', 'Unknown material'),
        'function': result.get('function', 'Unknown function'),
        'rarity': result.get('rarity', 'Unknown rarity')
    }
    
    # Ensure confidence is between 0 and 1
    validated['confidence'] = max(0.0, min(1.0, validated['confidence']))
    
    return validated


def get_artifact_suggestions(description):
    """
    Get artifact suggestions based on a text description
    Useful for providing alternative identifications
    """
    
    try:
        client = get_gemini_client()
        
        prompt = f"""
        Based on this description of an archaeological artifact: "{description}"
        
        Suggest 3-5 possible artifact types that might match this description.
        Respond with a JSON object containing a "suggestions" array of objects, each containing:
        - name: Artifact name
        - likelihood: How likely this identification is (0.0 to 1.0)
        - reason: Brief explanation of why this could be a match
        
        Focus on common archaeological artifacts that match the description.
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are an archaeological expert providing artifact identification suggestions.",
                response_mime_type="application/json",
            ),
        )
        
        if response.text:
            result = json.loads(response.text)
            return result.get('suggestions', [])
        else:
            return []
        
    except Exception as e:
        return []


def compare_with_reference(base64_image, reference_artifacts):
    """
    Compare an uploaded image with reference artifacts from the database
    Returns similarity scores and matches
    """
    
    try:
        client = get_gemini_client()
        
        # Convert base64 to bytes
        image_bytes = base64.b64decode(base64_image)
        
        # Create a description of reference artifacts for comparison
        reference_descriptions = []
        for artifact in reference_artifacts:
            reference_descriptions.append(f"- {artifact['name']}: {artifact['description']}")
        
        references_text = "\n".join(reference_descriptions)
        
        prompt = f"""
        Compare this artifact image with these reference artifacts from our database:
        
        {references_text}
        
        Respond with JSON containing:
        - closest_match: Name of the most similar reference artifact
        - similarity_score: How similar it is (0.0 to 1.0)
        - differences: Key differences from the closest match
        - alternative_matches: Array of other possible matches with their similarity scores
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        
        if response.text:
            result = json.loads(response.text)
            return result
        else:
            return None
        
    except Exception as e:
        return None
