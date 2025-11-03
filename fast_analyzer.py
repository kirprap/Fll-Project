"""
Fast AI Analyzer - Optimized for <2 minute response times
Provides multiple speed/quality tiers
"""

from typing import Dict, Any, Optional
import logging
from PIL import Image
import time

logger = logging.getLogger(__name__)


class FastAnalyzer:
    """
    Multi-tier analyzer with speed optimizations.
    
    Tiers:
    - INSTANT: ViT only (~1-2 seconds)
    - FAST: LLaVA 7B (~20-40 seconds)
    - BALANCED: Qwen2-VL 7B (~30-60 seconds)
    - QUALITY: Qwen3-VL latest (~1-2 minutes)
    """
    
    def __init__(self, tier: str = "FAST"):
        self.tier = tier.upper()
        self._setup_analyzer()
    
    def _setup_analyzer(self):
        """Setup the appropriate analyzer based on tier."""
        if self.tier == "INSTANT":
            # Use ViT from artifact_database
            from artifact_database import _vit_classify_image, _decode_base64_image_to_pil
            self.analyzer_type = "vit"
            self.expected_time = "1-2 seconds"
            
        elif self.tier in ["FAST", "BALANCED", "QUALITY"]:
            # Use Ollama with appropriate model
            from ai_analyzer import OllamaClient
            
            model_map = {
                "FAST": "llava:7b",           # ~20-40 seconds
                "BALANCED": "qwen2-vl:7b",    # ~30-60 seconds  
                "QUALITY": "qwen3-vl:latest"  # ~1-2 minutes
            }
            
            self.model = model_map[self.tier]
            self.analyzer_type = "ollama"
            self.ollama = OllamaClient(
                model=self.model,
                timeout=120  # 2 minutes max
            )
            
            time_map = {
                "FAST": "20-40 seconds",
                "BALANCED": "30-60 seconds",
                "QUALITY": "1-2 minutes"
            }
            self.expected_time = time_map[self.tier]
            
        else:
            raise ValueError(f"Unknown tier: {self.tier}. Use INSTANT, FAST, BALANCED, or QUALITY")
        
        logger.info(f"FastAnalyzer initialized: tier={self.tier}, expected_time={self.expected_time}")
    
    def analyze_artifact(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze an artifact image with timing.
        
        Returns:
            Dict with analysis results and timing info
        """
        start_time = time.time()
        
        try:
            if self.analyzer_type == "vit":
                result = self._analyze_with_vit(image)
            else:
                result = self._analyze_with_ollama(image)
            
            elapsed = time.time() - start_time
            result["analysis_time"] = f"{elapsed:.1f}s"
            result["tier"] = self.tier
            
            logger.info(f"Analysis complete: tier={self.tier}, time={elapsed:.1f}s")
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Analysis failed after {elapsed:.1f}s: {str(e)}")
            raise
    
    def _analyze_with_vit(self, image: Image.Image) -> Dict[str, Any]:
        """Fast ViT classification."""
        from artifact_database import _vit_classify_image
        
        label, confidence = _vit_classify_image(image)
        
        return {
            "name": label,
            "description": f"Classified as: {label}",
            "confidence": float(confidence),
            "method": "ViT Classification",
            "quality": "Basic"
        }
    
    def _analyze_with_ollama(self, image: Image.Image) -> Dict[str, Any]:
        """Ollama-based analysis with optimized prompt."""
        
        # Shorter, more focused prompt for faster response
        prompt = (
            "Briefly identify this artifact in 2-3 sentences: "
            "type, material, approximate age, and cultural origin."
        )
        
        description = self.ollama.generate(prompt, image=image).strip()
        
        # Extract name from first sentence
        name = description.split(".")[0] if description else "Unknown artifact"
        
        return {
            "name": name,
            "description": description,
            "confidence": 0.85,  # Ollama doesn't provide confidence
            "method": f"Ollama ({self.model})",
            "quality": self.tier
        }
    
    @staticmethod
    def get_available_tiers() -> Dict[str, str]:
        """Return available speed tiers and their expected times."""
        return {
            "INSTANT": "1-2 seconds (ViT only, basic classification)",
            "FAST": "20-40 seconds (LLaVA 7B, good quality)",
            "BALANCED": "30-60 seconds (Qwen2-VL 7B, better quality)",
            "QUALITY": "1-2 minutes (Qwen3-VL, best quality)"
        }
    
    @staticmethod
    def recommend_tier(max_wait_seconds: int) -> str:
        """Recommend a tier based on maximum acceptable wait time."""
        if max_wait_seconds <= 5:
            return "INSTANT"
        elif max_wait_seconds <= 45:
            return "FAST"
        elif max_wait_seconds <= 90:
            return "BALANCED"
        else:
            return "QUALITY"


def check_model_availability() -> Dict[str, bool]:
    """Check which Ollama models are available."""
    import subprocess
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        output = result.stdout.lower()
        
        return {
            "llava:7b": "llava:7b" in output or "llava" in output,
            "qwen2-vl:7b": "qwen2-vl:7b" in output or "qwen2-vl" in output,
            "qwen3-vl:latest": "qwen3-vl:latest" in output or "qwen3-vl" in output,
        }
    except Exception as e:
        logger.warning(f"Could not check model availability: {e}")
        return {}


def download_fast_model():
    """Download the recommended fast model (LLaVA 7B)."""
    import subprocess
    
    print("📥 Downloading LLaVA 7B model (~4GB)...")
    print("This will take 2-5 minutes depending on your internet speed.")
    print("")
    
    try:
        subprocess.run(
            ["ollama", "pull", "llava:7b"],
            check=True
        )
        print("✅ Model downloaded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download model: {e}")
        return False
    except FileNotFoundError:
        print("❌ Ollama not found. Please install Ollama first: https://ollama.ai")
        return False


if __name__ == "__main__":
    # Test the analyzer
    print("FastAnalyzer - Speed Tiers")
    print("=" * 50)
    
    tiers = FastAnalyzer.get_available_tiers()
    for tier, description in tiers.items():
        print(f"{tier:12} - {description}")
    
    print("\n" + "=" * 50)
    print("\nChecking model availability...")
    
    available = check_model_availability()
    for model, is_available in available.items():
        status = "✅ Available" if is_available else "❌ Not installed"
        print(f"{model:20} - {status}")
    
    print("\n" + "=" * 50)
    print("\nRecommendation for 2-minute target: Use FAST or BALANCED tier")
    print("\nTo download the fast model:")
    print("  ollama pull llava:7b")

