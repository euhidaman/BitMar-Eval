#!/usr/bin/env python3
"""
Quick verification script to test BitMar model loading before full evaluation
"""

import torch
from transformers import AutoModel, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_bitmar_loading():
    """Test if BitMar model can be loaded successfully"""
    try:
        logger.info("🔍 Testing BitMar model loading...")

        # Check GPU availability
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device("cpu")
            logger.info("⚠️  Using CPU (no CUDA available)")

        # Test model loading
        model_path = "euhidaman/bitmar-attention-multimodal"
        logger.info(f"📥 Loading model: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True
        )

        if device.type != "cuda":
            model = model.to(device)

        logger.info("✅ Model loaded successfully!")

        # Test tokenization
        test_text = "What is the capital of France?"
        inputs = tokenizer(test_text, return_tensors="pt").to(device)
        logger.info(f"✅ Tokenization test passed: {inputs['input_ids'].shape}")

        # Test inference
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        logger.info(f"✅ Generation test passed: '{response}'")

        logger.info("🎉 All tests passed! BitMar model is ready for evaluation.")
        return True

    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

if __name__ == "__main__":
    success = test_bitmar_loading()
    exit(0 if success else 1)
