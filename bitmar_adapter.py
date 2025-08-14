"""
BitMar Model Adapter for Evaluation
Handles the complex BitMar architecture for evaluation compatibility
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Union
import logging
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

class BitMarEvaluationAdapter:
    """Adapter to make BitMar model compatible with evaluation frameworks"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load BitMar model with proper handling"""
        try:
            # Try loading from checkpoint first
            if self.model_path.endswith('.pt') or self.model_path.endswith('.pth'):
                self._load_from_checkpoint()
            else:
                self._load_from_huggingface()

        except Exception as e:
            logger.error(f"Failed to load BitMar model: {e}")
            raise

    def _load_from_checkpoint(self):
        """Load model from local checkpoint"""
        checkpoint_path = Path(self.model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading BitMar model from checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Get config from checkpoint
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Use default config if not in checkpoint
            config = self._get_default_config()

        # Import BitMar model - fix the import path
        bitmar_src_path = str(Path(__file__).parent.parent / "BitMar" / "src")
        if bitmar_src_path not in sys.path:
            sys.path.insert(0, bitmar_src_path)

        try:
            from model import create_bitmar_model, BitMarModel
            logger.info("✅ Successfully imported BitMar model components")
        except ImportError as e:
            logger.error(f"Failed to import BitMar model: {e}")
            logger.error(f"Tried importing from: {bitmar_src_path}")
            raise ImportError(f"Could not import BitMar model. Make sure BitMar/src/model.py exists at {bitmar_src_path}")

        # Create model
        self.model = create_bitmar_model(config['model'])

        # Load state dict
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        # Get tokenizer from model
        self.tokenizer = self.model.tokenizer

        logger.info("✅ BitMar model loaded from checkpoint")

    def _load_from_huggingface(self):
        """Load model from Hugging Face Hub"""
        try:
            from transformers import AutoModel, AutoTokenizer, AutoConfig

            logger.info(f"Loading BitMar model from HF Hub: {self.model_path}")

            # Try to load config first
            try:
                config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
                logger.info("Config loaded from HF Hub")
            except:
                logger.warning("Could not load config from HF Hub, using default")
                config = self._get_default_config()

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True
            )

            if self.device.type != "cuda":
                self.model = self.model.to(self.device)

            self.model.eval()

            logger.info("✅ BitMar model loaded from HF Hub")

        except Exception as e:
            logger.error(f"Failed to load from HF Hub: {e}")
            # Fallback to checkpoint loading if HF fails
            checkpoint_path = Path("checkpoints_100M_dataset") / "latest_checkpoint.pt"
            if checkpoint_path.exists():
                logger.info("Falling back to local checkpoint")
                self.model_path = str(checkpoint_path)
                self._load_from_checkpoint()
            else:
                raise RuntimeError(f"Could not load model from {self.model_path} and no local checkpoint found")

    def _get_default_config(self):
        """Get default BitMar configuration"""
        return {
            'model': {
                'vocab_size': 50257,
                'text_encoder_dim': 128,
                'text_encoder_layers': 4,
                'text_encoder_heads': 4,
                'text_decoder_dim': 128,
                'text_decoder_layers': 4,
                'text_decoder_heads': 4,
                'vision_encoder_dim': 768,
                'vision_latent_size': 128,
                'vision_hidden_size': 64,
                'fusion_hidden_size': 128,
                'fusion_num_heads': 4,
                'fusion_num_layers': 2,
                'memory_size': 32,
                'episode_dim': 128,
                'memory_alpha': 0.2,
                'direct_writing': True,
                'max_seq_len': 256,
                'dropout': 0.15
            }
        }

    def generate_response(self, prompt: str, max_length: int = 100, temperature: float = 0.1) -> str:
        """Generate response using BitMar model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            # Create dummy vision features (all zeros for text-only evaluation)
            batch_size = inputs['input_ids'].shape[0]
            vision_features = torch.zeros(batch_size, 768, device=self.device)

            # Generate using BitMar's generate method
            if hasattr(self.model, 'generate') and callable(self.model.generate):
                # Use BitMar's built-in generate method
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    vision_features=vision_features,
                    max_length=max_length,
                    temperature=temperature
                )

                # Extract generated text
                if isinstance(outputs, dict) and 'generated_text' in outputs:
                    return outputs['generated_text'][0] if isinstance(outputs['generated_text'], list) else outputs['generated_text']
                elif isinstance(outputs, dict) and 'generated_ids' in outputs:
                    generated_ids = outputs['generated_ids'][0]
                    # Remove input tokens
                    new_tokens = generated_ids[inputs['input_ids'].shape[1]:]
                    return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Fallback: use model forward pass for generation
            return self._generate_with_forward_pass(inputs, vision_features, max_length, temperature)

        except Exception as e:
            logger.warning(f"Generation failed for prompt: {prompt[:50]}... Error: {e}")
            return ""

    def _generate_with_forward_pass(self, inputs, vision_features, max_length, temperature):
        """Fallback generation using forward pass"""
        try:
            with torch.no_grad():
                # Get model outputs
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    vision_features=vision_features,
                    mode="inference"
                )

                # Get logits
                logits = outputs['logits']

                # Simple greedy decoding for the last token
                next_token_logits = logits[:, -1, :] / temperature
                next_token = torch.argmax(next_token_logits, dim=-1)

                # Decode the token
                response = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                return response.strip()

        except Exception as e:
            logger.warning(f"Forward pass generation failed: {e}")
            return ""

    def evaluate_multiple_choice(self, prompt: str, choices: List[str]) -> int:
        """Evaluate multiple choice by comparing perplexities"""
        try:
            perplexities = []

            for choice in choices:
                full_text = f"{prompt}\nAnswer: {choice}"

                # Tokenize
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                # Create dummy vision features
                batch_size = inputs['input_ids'].shape[0]
                vision_features = torch.zeros(batch_size, 768, device=self.device)

                # Calculate perplexity using model
                with torch.no_grad():
                    try:
                        outputs = self.model(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            vision_features=vision_features,
                            labels=inputs['input_ids'],
                            mode="inference"
                        )

                        if 'loss' in outputs and outputs['loss'] is not None:
                            loss = outputs['loss']
                            perplexity = torch.exp(loss).item()
                        else:
                            # Fallback: compute loss from logits
                            logits = outputs['logits']
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = inputs['input_ids'][..., 1:].contiguous()
                            loss = nn.CrossEntropyLoss()(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1)
                            )
                            perplexity = torch.exp(loss).item()

                        perplexities.append(perplexity)

                    except Exception as e:
                        logger.warning(f"Error computing perplexity for choice '{choice}': {e}")
                        perplexities.append(float('inf'))

            # Return index of choice with lowest perplexity
            return perplexities.index(min(perplexities))

        except Exception as e:
            logger.warning(f"Multiple choice evaluation failed: {e}")
            return 0  # Default to first choice

    def get_model_info(self) -> Dict:
        """Get model information"""
        info = {
            'model_path': self.model_path,
            'device': str(self.device),
            'model_type': 'BitMar',
            'architecture': 'BitNet-quantized Vision-Language Episodic Memory Transformer'
        }

        if self.model:
            try:
                # Get parameter count
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

                info.update({
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'model_device': str(next(self.model.parameters()).device)
                })

                # Get config if available
                if hasattr(self.model, 'config'):
                    info['config'] = self.model.config

            except Exception as e:
                logger.warning(f"Could not get model info: {e}")

        return info


def create_bitmar_adapter(model_path: str, device: str = "cuda") -> BitMarEvaluationAdapter:
    """Create BitMar evaluation adapter"""
    return BitMarEvaluationAdapter(model_path, device)
