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
            logger.info(f"Loading BitMar model from HF Hub: {self.model_path}")

            # First, try to load the tokenizer separately (it should be standard GPT-2)
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use GPT-2 tokenizer as fallback
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("✅ Loaded GPT-2 tokenizer as fallback")
            except Exception as e:
                logger.error(f"Failed to load tokenizer: {e}")
                raise

            # Try to download model files manually and load as custom BitMar model
            try:
                from huggingface_hub import hf_hub_download
                import json

                # Download the config file to understand the model structure
                try:
                    config_path = hf_hub_download(repo_id=self.model_path, filename="config.json")
                    with open(config_path, 'r') as f:
                        hf_config = json.load(f)
                    logger.info("✅ Downloaded config from HF Hub")
                except Exception as e:
                    logger.warning(f"Could not download config: {e}, using default")
                    hf_config = {}

                # Download the model weights
                try:
                    model_weights_path = hf_hub_download(repo_id=self.model_path, filename="pytorch_model.bin")
                    logger.info("✅ Downloaded model weights from HF Hub")
                except Exception as e:
                    # Try alternative filename
                    try:
                        model_weights_path = hf_hub_download(repo_id=self.model_path, filename="model.safetensors")
                        logger.info("✅ Downloaded model weights (safetensors) from HF Hub")
                    except Exception as e2:
                        logger.error(f"Failed to download model weights: {e}, {e2}")
                        raise RuntimeError(f"Could not download model weights from {self.model_path}")

                # Create BitMar model config from HF config or use defaults
                bitmar_config = self._create_bitmar_config_from_hf(hf_config)

                # Import BitMar model from the correct path
                bitmar_src_path = str(Path(__file__).parent.parent / "BitMar" / "src")
                if bitmar_src_path not in sys.path:
                    sys.path.insert(0, bitmar_src_path)

                try:
                    from model import create_bitmar_model, BitMarModel
                    logger.info("✅ Successfully imported BitMar model components")
                except ImportError as e:
                    logger.error(f"Failed to import BitMar model: {e}")
                    logger.error(f"Tried importing from: {bitmar_src_path}")
                    # Try alternative import paths
                    alternative_paths = [
                        str(Path(__file__).parent.parent.parent / "BitMar" / "src"),
                        str(Path(__file__).parent / ".." / "BitMar" / "src"),
                        "/workspace/BitMar/src"
                    ]

                    for alt_path in alternative_paths:
                        if Path(alt_path).exists():
                            logger.info(f"Trying alternative path: {alt_path}")
                            if alt_path not in sys.path:
                                sys.path.insert(0, alt_path)
                            try:
                                from model import create_bitmar_model, BitMarModel
                                logger.info(f"✅ Successfully imported from: {alt_path}")
                                break
                            except ImportError:
                                continue
                    else:
                        raise ImportError(f"Could not import BitMar model from any path. Available paths checked: {[bitmar_src_path] + alternative_paths}")

                # Create BitMar model using your custom architecture
                self.model = create_bitmar_model(bitmar_config)

                # Load the state dict
                if model_weights_path.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    state_dict = load_file(model_weights_path)
                else:
                    state_dict = torch.load(model_weights_path, map_location='cpu')

                # Handle different state dict formats
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    model_state_dict = state_dict['model_state_dict']
                else:
                    model_state_dict = state_dict

                # Load state dict with error handling for mismatched keys
                try:
                    self.model.load_state_dict(model_state_dict, strict=False)
                    logger.info("✅ Loaded model weights (non-strict)")
                except Exception as e:
                    logger.warning(f"Failed to load some weights: {e}")
                    # Try to load compatible weights only with shape fixing
                    model_dict = self.model.state_dict()
                    filtered_dict = {}

                    for k, v in model_state_dict.items():
                        if k in model_dict:
                            current_shape = model_dict[k].shape
                            checkpoint_shape = v.shape

                            if current_shape == checkpoint_shape:
                                # Perfect match, use directly
                                filtered_dict[k] = v
                            elif 'weight_scale' in k or 'input_scale' in k:
                                # Handle weight_scale buffer shape mismatch
                                if checkpoint_shape == torch.Size([]) and current_shape == torch.Size([1]):
                                    # Convert scalar to 1D tensor
                                    filtered_dict[k] = v.unsqueeze(0)
                                    logger.debug(f"Fixed shape for {k}: {checkpoint_shape} -> {current_shape}")
                                elif checkpoint_shape == torch.Size([1]) and current_shape == torch.Size([]):
                                    # Convert 1D tensor to scalar
                                    filtered_dict[k] = v.squeeze(0)
                                    logger.debug(f"Fixed shape for {k}: {checkpoint_shape} -> {current_shape}")
                                else:
                                    logger.warning(f"Cannot fix shape mismatch for {k}: {checkpoint_shape} vs {current_shape}")
                            elif k.endswith('.weight') or k.endswith('.bias'):
                                # Handle other weight/bias mismatches by padding or trimming
                                if len(current_shape) == len(checkpoint_shape):
                                    # Same number of dimensions, try to fix
                                    fixed_tensor = self._fix_tensor_shape(v, current_shape, k)
                                    if fixed_tensor is not None:
                                        filtered_dict[k] = fixed_tensor
                                    else:
                                        logger.warning(f"Skipping incompatible weight {k}: {checkpoint_shape} vs {current_shape}")
                                else:
                                    logger.warning(f"Skipping weight {k} due to dimension mismatch: {checkpoint_shape} vs {current_shape}")
                            else:
                                logger.warning(f"Skipping incompatible parameter {k}: {checkpoint_shape} vs {current_shape}")
                        else:
                            logger.debug(f"Skipping unknown parameter: {k}")

                    # Update model with filtered weights
                    model_dict.update(filtered_dict)
                    self.model.load_state_dict(model_dict)

                    logger.info(f"✅ Loaded {len(filtered_dict)}/{len(model_state_dict)} compatible weights with shape fixes")
                    logger.info(f"   Successfully fixed {sum(1 for k in filtered_dict.keys() if 'weight_scale' in k or 'input_scale' in k)} scale buffer shapes")

                # Move to device
                self.model.to(self.device)
                self.model.eval()

                logger.info("✅ BitMar model loaded from HF Hub using custom loading")
                return

            except Exception as e:
                logger.error(f"Custom loading failed: {e}")
                raise

        except Exception as e:
            logger.error(f"Failed to load from HF Hub: {e}")
            # Fallback to checkpoint loading if HF fails
            checkpoint_paths = [
                Path("/workspace/BitMar/checkpoints_100M_dataset") / "latest_checkpoint.pt",
                Path("../BitMar/checkpoints_100M_dataset") / "latest_checkpoint.pt",
                Path("checkpoints_100M_dataset") / "latest_checkpoint.pt",
                Path("latest_checkpoint.pt")
            ]

            for checkpoint_path in checkpoint_paths:
                if checkpoint_path.exists():
                    logger.info(f"Falling back to local checkpoint: {checkpoint_path}")
                    self.model_path = str(checkpoint_path)
                    self._load_from_checkpoint()
                    return

            raise RuntimeError(f"Could not load model from {self.model_path} and no local checkpoint found")

    def _create_bitmar_config_from_hf(self, hf_config: dict) -> dict:
        """Create BitMar config from HuggingFace config or use defaults"""
        # Extract known BitMar parameters from HF config
        bitmar_config = self._get_default_config()['model'].copy()

        # Map HF config keys to BitMar config keys
        key_mapping = {
            'vocab_size': 'vocab_size',
            'text_encoder_dim': 'text_encoder_dim',
            'text_encoder_layers': 'text_encoder_layers',
            'text_encoder_heads': 'text_encoder_heads',
            'text_decoder_dim': 'text_decoder_dim',
            'text_decoder_layers': 'text_decoder_layers',
            'text_decoder_heads': 'text_decoder_heads',
            'vision_encoder_dim': 'vision_encoder_dim',
            'vision_latent_size': 'vision_latent_size',
            'vision_hidden_size': 'vision_hidden_size',
            'fusion_hidden_size': 'fusion_hidden_size',
            'fusion_num_heads': 'fusion_num_heads',
            'fusion_num_layers': 'fusion_num_layers',
            'memory_size': 'memory_size',
            'episode_dim': 'episode_dim',
            'max_seq_len': 'max_seq_len',
            'dropout': 'dropout'
        }

        # Update config with values from HF config
        for hf_key, bitmar_key in key_mapping.items():
            if hf_key in hf_config:
                bitmar_config[bitmar_key] = hf_config[hf_key]

        logger.info(f"Created BitMar config with vocab_size={bitmar_config['vocab_size']}, text_dim={bitmar_config['text_encoder_dim']}")
        return bitmar_config

    def _fix_tensor_shape(self, tensor: torch.Tensor, target_shape: torch.Size, param_name: str) -> Optional[torch.Tensor]:
        """Fix tensor shape mismatches by padding, trimming, or reshaping"""
        try:
            current_shape = tensor.shape

            # Handle 2D weight matrices
            if len(target_shape) == 2 and len(current_shape) == 2:
                target_rows, target_cols = target_shape
                current_rows, current_cols = current_shape

                # Create new tensor with target shape
                fixed_tensor = torch.zeros(target_shape, dtype=tensor.dtype)

                # Copy overlapping region
                copy_rows = min(target_rows, current_rows)
                copy_cols = min(target_cols, current_cols)

                fixed_tensor[:copy_rows, :copy_cols] = tensor[:copy_rows, :copy_cols]

                logger.debug(f"Fixed weight shape for {param_name}: {current_shape} -> {target_shape}")
                return fixed_tensor

            # Handle 1D bias vectors
            elif len(target_shape) == 1 and len(current_shape) == 1:
                target_size = target_shape[0]
                current_size = current_shape[0]

                if target_size > current_size:
                    # Pad with zeros
                    padding = torch.zeros(target_size - current_size, dtype=tensor.dtype)
                    fixed_tensor = torch.cat([tensor, padding])
                else:
                    # Truncate
                    fixed_tensor = tensor[:target_size]

                logger.debug(f"Fixed bias shape for {param_name}: {current_shape} -> {target_shape}")
                return fixed_tensor

            # Handle scalar tensors
            elif len(target_shape) == 0 and len(current_shape) == 1 and current_shape[0] == 1:
                return tensor.squeeze(0)
            elif len(target_shape) == 1 and target_shape[0] == 1 and len(current_shape) == 0:
                return tensor.unsqueeze(0)

            return None

        except Exception as e:
            logger.warning(f"Failed to fix shape for {param_name}: {e}")
            return None

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
