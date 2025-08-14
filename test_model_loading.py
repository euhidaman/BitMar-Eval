#!/usr/bin/env python3
"""
Quick verification script to test BitMar model loading before full evaluation
"""

import torch
import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from bitmar_adapter import create_bitmar_adapter
    BITMAR_ADAPTER_AVAILABLE = True
except ImportError:
    BITMAR_ADAPTER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_bitmar_loading():
    """Test if BitMar model can be loaded successfully"""
    try:
        logger.info("🔍 Testing BitMar model loading...")

        if not BITMAR_ADAPTER_AVAILABLE:
            logger.error("❌ BitMar adapter not available. Make sure bitmar_adapter.py is in the same directory.")
            return False

        # Check GPU availability
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device("cpu")
            logger.info("⚠️  Using CPU (no CUDA available)")

        # Test model loading using BitMar adapter
        model_path = "euhidaman/bitmar-attention-multimodal"
        logger.info(f"📥 Loading model: {model_path}")

        # Create BitMar adapter
        adapter = create_bitmar_adapter(model_path, str(device))

        logger.info("✅ Model loaded successfully!")

        # Get model info
        model_info = adapter.get_model_info()
        logger.info(f"Model type: {model_info.get('model_type', 'Unknown')}")
        logger.info(f"Architecture: {model_info.get('architecture', 'Unknown')}")
        if 'total_parameters' in model_info:
            logger.info(f"Total parameters: {model_info['total_parameters']:,}")

        # Test tokenization
        test_text = "What is the capital of France?"
        logger.info(f"🧪 Testing generation with: '{test_text}'")

        # Test generation
        response = adapter.generate_response(test_text, max_length=20, temperature=0.1)
        logger.info(f"✅ Generation test passed: '{response}'")

        # Test multiple choice evaluation
        logger.info("🧪 Testing multiple choice evaluation...")
        prompt = "What is the capital of France?"
        choices = ["London", "Paris", "Berlin", "Madrid"]
        predicted_idx = adapter.evaluate_multiple_choice(prompt, choices)
        logger.info(f"✅ Multiple choice test passed: predicted choice {predicted_idx} ({choices[predicted_idx]})")

        logger.info("🎉 All tests passed! BitMar model is ready for evaluation.")
        return True

    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bitmar_loading()
    exit(0 if success else 1)
