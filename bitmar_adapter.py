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
import os
import numpy as np

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
        """Evaluate multiple choice by comparing perplexities with maximum CUDA safety"""
        try:
            # Force CUDA debugging mode for better error reporting
            if self.device.type == 'cuda':
                os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            perplexities = []

            # Enhanced input validation
            if not choices or len(choices) == 0:
                logger.warning("Empty choices list provided")
                return 0

            if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
                logger.warning("Invalid or empty prompt provided")
                return 0

            # Aggressive CUDA memory management
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Check available memory
                try:
                    memory_allocated = torch.cuda.memory_allocated(self.device)
                    memory_reserved = torch.cuda.memory_reserved(self.device)
                    logger.debug(f"CUDA memory - Allocated: {memory_allocated/1024**2:.1f}MB, Reserved: {memory_reserved/1024**2:.1f}MB")
                except Exception as mem_error:
                    logger.warning(f"Could not check CUDA memory: {mem_error}")

            # Process each choice with maximum safety
            for i, choice in enumerate(choices):
                try:
                    # Comprehensive choice validation
                    if choice is None:
                        logger.warning(f"Choice {i} is None")
                        perplexities.append(float('inf'))
                        continue

                    if not isinstance(choice, str):
                        logger.warning(f"Choice {i} is not a string: {type(choice)}")
                        choice = str(choice)

                    choice = choice.strip()
                    if len(choice) == 0:
                        logger.warning(f"Choice {i} is empty after stripping")
                        perplexities.append(float('inf'))
                        continue

                    # Safe text preparation with extensive error handling
                    try:
                        # Clean and validate text encoding
                        prompt_clean = prompt.encode('utf-8', errors='replace').decode('utf-8')
                        choice_clean = choice.encode('utf-8', errors='replace').decode('utf-8')

                        # Remove potentially problematic characters
                        import re
                        prompt_clean = re.sub(r'[^\x00-\x7F]+', ' ', prompt_clean)
                        choice_clean = re.sub(r'[^\x00-\x7F]+', ' ', choice_clean)

                        # Create safe text with length limits
                        full_text = f"{prompt_clean[:1000]}\nAnswer: {choice_clean[:200]}"

                        # Ensure reasonable length
                        if len(full_text) > 1500:
                            full_text = full_text[:1500]

                    except Exception as text_error:
                        logger.warning(f"Text preparation failed for choice {i}: {text_error}")
                        perplexities.append(float('inf'))
                        continue

                    # Ultra-safe tokenization with multiple fallbacks
                    inputs = None
                    try:
                        # Strategy 1: Safe tokenization with explicit parameters
                        inputs = self.tokenizer(
                            full_text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=256,  # Reduced from 512 for safety
                            padding=False,
                            add_special_tokens=True,
                            return_attention_mask=True,
                            return_token_type_ids=False,
                            return_offsets_mapping=False
                        )

                        # Validate tokenization immediately
                        if not inputs or 'input_ids' not in inputs:
                            raise ValueError("Tokenization returned invalid result")

                        if inputs['input_ids'].numel() == 0:
                            raise ValueError("Tokenization produced empty tensor")

                        if inputs['input_ids'].dim() != 2:
                            raise ValueError(f"Invalid input_ids dimensions: {inputs['input_ids'].shape}")

                        if inputs['input_ids'].size(1) == 0:
                            raise ValueError("Tokenization produced zero-length sequence")

                    except Exception as tokenize_error:
                        logger.warning(f"Primary tokenization failed for choice {i}: {tokenize_error}")

                        # Strategy 2: Minimal tokenization
                        try:
                            simple_text = choice_clean[:100]  # Very short text
                            inputs = self.tokenizer(
                                simple_text,
                                return_tensors="pt",
                                truncation=True,
                                max_length=64,
                                padding=False,
                                add_special_tokens=True
                            )

                            if inputs['input_ids'].numel() == 0:
                                raise ValueError("Simple tokenization failed")

                            logger.debug(f"Used simplified tokenization for choice {i}")

                        except Exception as simple_error:
                            logger.warning(f"Simplified tokenization failed for choice {i}: {simple_error}")

                            # Strategy 3: Emergency manual tokenization
                            try:
                                # Use just a few safe tokens
                                safe_tokens = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]
                                inputs = {
                                    'input_ids': torch.tensor([safe_tokens], dtype=torch.long),
                                    'attention_mask': torch.ones((1, len(safe_tokens)), dtype=torch.long)
                                }
                                logger.debug(f"Used emergency tokenization for choice {i}")

                            except Exception as manual_error:
                                logger.error(f"All tokenization failed for choice {i}: {manual_error}")
                                perplexities.append(float('inf'))
                                continue

                    # Extensive token validation and sanitization
                    try:
                        vocab_size = getattr(self.tokenizer, 'vocab_size', 50257)
                        input_ids = inputs['input_ids']

                        # Multiple validation checks
                        if not torch.is_tensor(input_ids):
                            raise ValueError("input_ids is not a tensor")

                        if input_ids.numel() == 0:
                            raise ValueError("input_ids tensor is empty")

                        if input_ids.dim() != 2:
                            raise ValueError(f"input_ids has wrong dimensions: {input_ids.shape}")

                        if input_ids.size(0) != 1:
                            raise ValueError(f"input_ids batch size should be 1, got {input_ids.size(0)}")

                        if input_ids.size(1) == 0:
                            raise ValueError("input_ids sequence length is 0")

                        # Check for valid token range with detailed logging
                        try:
                            min_token = input_ids.min().item()
                            max_token = input_ids.max().item()

                            # Log token statistics for debugging
                            logger.debug(f"Choice {i}: tokens range [{min_token}, {max_token}], vocab_size={vocab_size}")

                            if min_token < 0:
                                logger.warning(f"Negative token ID found: {min_token}")
                                input_ids = torch.clamp(input_ids, min=0)

                            if max_token >= vocab_size:
                                logger.warning(f"Token ID exceeds vocab size: {max_token} >= {vocab_size}")
                                input_ids = torch.clamp(input_ids, max=vocab_size - 1)

                            # Update inputs with clamped values
                            inputs['input_ids'] = input_ids

                        except Exception as bounds_error:
                            logger.warning(f"Token bounds checking failed for choice {i}: {bounds_error}")
                            # Create completely safe fallback tokens
                            safe_length = min(10, inputs['input_ids'].size(1))
                            safe_tokens = torch.full((1, safe_length), self.tokenizer.eos_token_id, dtype=torch.long)
                            inputs['input_ids'] = safe_tokens
                            inputs['attention_mask'] = torch.ones((1, safe_length), dtype=torch.long)

                        # Ensure attention mask compatibility
                        if 'attention_mask' not in inputs:
                            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
                        elif inputs['attention_mask'].shape != inputs['input_ids'].shape:
                            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

                        # Final tensor validation
                        for key, tensor in inputs.items():
                            if torch.is_tensor(tensor):
                                if not torch.isfinite(tensor).all():
                                    logger.warning(f"Non-finite values in {key} for choice {i}")
                                    if key == 'input_ids':
                                        tensor.fill_(self.tokenizer.eos_token_id)
                                    else:
                                        tensor.fill_(1)

                    except Exception as validation_error:
                        logger.warning(f"Token validation failed for choice {i}: {validation_error}")
                        perplexities.append(float('inf'))
                        continue

                    # Ultra-safe device transfer
                    try:
                        device_inputs = {}
                        for key, value in inputs.items():
                            if torch.is_tensor(value):
                                try:
                                    # Force blocking transfer and validate
                                    transferred = value.to(self.device, non_blocking=False)

                                    # Verify transfer worked
                                    if transferred.device != self.device:
                                        logger.warning(f"Device transfer failed for {key}, using CPU")
                                        transferred = value.cpu()

                                    device_inputs[key] = transferred

                                except Exception as transfer_error:
                                    logger.warning(f"Failed to transfer {key} to device: {transfer_error}")
                                    device_inputs[key] = value.cpu()
                            else:
                                device_inputs[key] = value

                        inputs = device_inputs

                        # CUDA synchronization and error check
                        if self.device.type == 'cuda':
                            try:
                                torch.cuda.synchronize(self.device)
                                # Small test operation to ensure CUDA is working
                                test_tensor = torch.tensor([1.0], device=self.device)
                                test_result = test_tensor + 1.0
                                del test_tensor, test_result
                                torch.cuda.synchronize(self.device)
                            except Exception as cuda_error:
                                logger.error(f"CUDA synchronization/test failed for choice {i}: {cuda_error}")
                                torch.cuda.empty_cache()
                                perplexities.append(float('inf'))
                                continue

                    except Exception as device_error:
                        logger.warning(f"Device operations failed for choice {i}: {device_error}")
                        perplexities.append(float('inf'))
                        continue

                    # Safe vision features creation with validation
                    try:
                        batch_size = inputs['input_ids'].size(0)

                        # Create vision features on CPU first, then transfer
                        vision_features_cpu = torch.zeros(batch_size, 768, dtype=torch.float32)

                        # Transfer to device safely
                        if self.device.type == 'cuda':
                            try:
                                vision_features = vision_features_cpu.to(self.device, non_blocking=False)
                                torch.cuda.synchronize(self.device)
                            except Exception as vision_transfer_error:
                                logger.warning(f"Vision features transfer failed: {vision_transfer_error}")
                                vision_features = vision_features_cpu  # Use CPU version
                        else:
                            vision_features = vision_features_cpu

                        # Validate vision features
                        if not torch.isfinite(vision_features).all():
                            logger.warning(f"Non-finite vision features for choice {i}")
                            vision_features.fill_(0.0)

                    except Exception as vision_error:
                        logger.warning(f"Vision features creation failed for choice {i}: {vision_error}")
                        perplexities.append(float('inf'))
                        continue

                    # Model inference with maximum safety
                    try:
                        # Ensure model is in eval mode
                        if hasattr(self.model, 'eval'):
                            self.model.eval()

                        # Clear any gradients
                        if hasattr(self.model, 'zero_grad'):
                            self.model.zero_grad()

                        # Prepare model inputs with comprehensive validation
                        model_inputs = {
                            'input_ids': inputs['input_ids'],
                            'attention_mask': inputs['attention_mask'],
                            'vision_features': vision_features,
                            'labels': inputs['input_ids'].clone().detach(),
                            'mode': "inference"
                        }

                        # Validate all model inputs
                        for key, value in model_inputs.items():
                            if torch.is_tensor(value):
                                if not torch.isfinite(value).all():
                                    logger.error(f"Non-finite values in model input {key} for choice {i}")
                                    raise ValueError(f"Non-finite values in {key}")

                                if value.numel() == 0:
                                    logger.error(f"Empty tensor in model input {key} for choice {i}")
                                    raise ValueError(f"Empty tensor in {key}")

                        # Model forward pass with comprehensive error handling
                        perplexity = float('inf')
                        try:
                            with torch.no_grad():
                                # Set autocast for mixed precision safety
                                if self.device.type == 'cuda':
                                    with torch.cuda.amp.autocast(enabled=False):  # Disable autocast for safety
                                        outputs = self.model(**model_inputs)
                                else:
                                    outputs = self.model(**model_inputs)

                                # Validate model outputs immediately
                                if outputs is None:
                                    logger.warning(f"Model returned None for choice {i}")
                                    perplexity = float('inf')
                                elif not isinstance(outputs, dict):
                                    logger.warning(f"Model returned non-dict output for choice {i}")
                                    perplexity = float('inf')
                                elif 'loss' in outputs and outputs['loss'] is not None:
                                    loss = outputs['loss']

                                    # Comprehensive loss validation
                                    if not torch.is_tensor(loss):
                                        logger.warning(f"Loss is not a tensor for choice {i}")
                                        perplexity = float('inf')
                                    elif loss.numel() != 1:
                                        logger.warning(f"Loss tensor has wrong size for choice {i}: {loss.shape}")
                                        perplexity = float('inf')
                                    elif not torch.isfinite(loss):
                                        logger.warning(f"Non-finite loss for choice {i}: {loss}")
                                        perplexity = float('inf')
                                    elif loss.item() < 0:
                                        logger.warning(f"Negative loss for choice {i}: {loss.item()}")
                                        perplexity = float('inf')
                                    else:
                                        # Safe perplexity computation with multiple safeguards
                                        try:
                                            loss_value = loss.item()

                                            # Clamp loss to prevent overflow
                                            loss_clamped = min(max(loss_value, 0.0), 15.0)

                                            # Compute perplexity safely
                                            perplexity = math.exp(loss_clamped)

                                            # Final perplexity validation
                                            if not math.isfinite(perplexity) or perplexity <= 0:
                                                logger.warning(f"Invalid perplexity for choice {i}: {perplexity}")
                                                perplexity = float('inf')
                                            elif perplexity > 1e6:  # Reasonable upper bound
                                                logger.warning(f"Extremely high perplexity for choice {i}: {perplexity}")
                                                perplexity = 1e6

                                        except Exception as perp_error:
                                            logger.warning(f"Perplexity computation failed for choice {i}: {perp_error}")
                                            perplexity = float('inf')
                                else:
                                    # Fallback: compute loss from logits with maximum safety
                                    if 'logits' in outputs and outputs['logits'] is not None:
                                        try:
                                            logits = outputs['logits']

                                            # Validate logits
                                            if not torch.is_tensor(logits):
                                                logger.warning(f"Logits not a tensor for choice {i}")
                                                perplexity = float('inf')
                                            elif logits.numel() == 0:
                                                logger.warning(f"Empty logits for choice {i}")
                                                perplexity = float('inf')
                                            elif not torch.isfinite(logits).all():
                                                logger.warning(f"Non-finite logits for choice {i}")
                                                perplexity = float('inf')
                                            else:
                                                # Safe loss computation from logits
                                                try:
                                                    shift_logits = logits[..., :-1, :].contiguous()
                                                    shift_labels = inputs['input_ids'][..., 1:].contiguous()

                                                    # Validate shapes
                                                    if shift_logits.size(0) != shift_labels.size(0):
                                                        logger.warning(f"Batch size mismatch in loss computation for choice {i}")
                                                        perplexity = float('inf')
                                                    elif shift_logits.size(1) != shift_labels.size(1):
                                                        logger.warning(f"Sequence length mismatch in loss computation for choice {i}")
                                                        perplexity = float('inf')
                                                    else:
                                                        # Ultra-safe loss computation
                                                        try:
                                                            loss_fn = torch.nn.CrossEntropyLoss(
                                                                ignore_index=-100,
                                                                reduction='mean',
                                                                label_smoothing=0.0
                                                            )

                                                            computed_loss = loss_fn(
                                                                shift_logits.view(-1, shift_logits.size(-1)),
                                                                shift_labels.view(-1)
                                                            )

                                                            if torch.isfinite(computed_loss) and computed_loss.item() >= 0:
                                                                loss_clamped = min(computed_loss.item(), 15.0)
                                                                perplexity = math.exp(loss_clamped)

                                                                # Final validation
                                                                if not math.isfinite(perplexity) or perplexity <= 0:
                                                                    perplexity = float('inf')
                                                            else:
                                                                logger.warning(f"Invalid computed loss for choice {i}: {computed_loss}")
                                                                perplexity = float('inf')

                                                        except Exception as loss_comp_error:
                                                            logger.warning(f"Loss computation from logits failed for choice {i}: {loss_comp_error}")
                                                            perplexity = float('inf')

                                                except Exception as logits_processing_error:
                                                    logger.warning(f"Logits processing failed for choice {i}: {logits_processing_error}")
                                                    perplexity = float('inf')
                                        except Exception as logits_error:
                                            logger.warning(f"Logits handling failed for choice {i}: {logits_error}")
                                            perplexity = float('inf')
                                    else:
                                        logger.warning(f"No loss or logits in model output for choice {i}")
                                        perplexity = float('inf')

                                # Store the computed perplexity
                                perplexities.append(perplexity)

                        except RuntimeError as runtime_error:
                            error_str = str(runtime_error).lower()
                            if any(cuda_term in error_str for cuda_term in ['cuda', 'device-side assert', 'gpu', 'device']):
                                logger.error(f"CUDA runtime error for choice {i}: {runtime_error}")
                                logger.error(f"This is likely a tensor indexing or memory access issue")

                                # Emergency CUDA recovery
                                try:
                                    if self.device.type == 'cuda':
                                        torch.cuda.empty_cache()
                                        torch.cuda.synchronize()
                                        torch.cuda.reset_peak_memory_stats()

                                except Exception as cleanup_error:
                                    logger.error(f"CUDA cleanup failed: {cleanup_error}")

                                perplexities.append(float('inf'))

                                # Skip remaining choices if we have severe CUDA errors
                                if "device-side assert" in error_str:
                                    logger.error("Device-side assert detected - aborting remaining choices for safety")
                                    perplexities.extend([float('inf')] * (len(choices) - i - 1))
                                    break
                            else:
                                logger.error(f"Runtime error for choice {i}: {runtime_error}")
                                perplexities.append(float('inf'))

                        except Exception as inference_error:
                            logger.warning(f"Model inference failed for choice {i}: {inference_error}")
                            perplexities.append(float('inf'))

                    except Exception as model_setup_error:
                        logger.error(f"Model setup failed for choice {i}: {model_setup_error}")
                        perplexities.append(float('inf'))

                except Exception as choice_error:
                    logger.error(f"Processing completely failed for choice {i}: {choice_error}")
                    perplexities.append(float('inf'))

                # Aggressive cleanup after each choice
                try:
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    # Force garbage collection
                    import gc
                    gc.collect()

                except Exception as cleanup_error:
                    logger.warning(f"Cleanup failed after choice {i}: {cleanup_error}")

            # Final result selection with extensive validation
            try:
                if not perplexities:
                    logger.warning("No perplexities computed, returning default choice 0")
                    return 0

                if all(p == float('inf') for p in perplexities):
                    logger.warning("All choices resulted in infinite perplexity, returning default choice 0")
                    return 0

                # Find best choice among finite perplexities
                finite_perplexities = []
                for i, p in enumerate(perplexities):
                    if math.isfinite(p) and p > 0:
                        finite_perplexities.append((i, p))

                if not finite_perplexities:
                    logger.warning("No valid finite perplexities found, returning default choice 0")
                    return 0

                # Select choice with minimum perplexity
                best_idx, best_perplexity = min(finite_perplexities, key=lambda x: x[1])

                logger.debug(f"Selected choice {best_idx} with perplexity {best_perplexity:.4f}")
                return best_idx

            except Exception as selection_error:
                logger.error(f"Choice selection failed: {selection_error}")
                return 0

        except Exception as e:
            logger.error(f"Multiple choice evaluation completely failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")

            # Emergency cleanup
            try:
                if hasattr(self, 'device') and self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Force cleanup
                import gc
                gc.collect()

            except Exception as final_cleanup_error:
                logger.error(f"Final cleanup failed: {final_cleanup_error}")

            return 0  # Default to first choice

        finally:
            # Always cleanup and reset CUDA settings
            try:
                if hasattr(self, 'device') and self.device.type == 'cuda':
                    torch.cuda.empty_cache()

                # Reset CUDA debugging settings
                if 'CUDA_LAUNCH_BLOCKING' in os.environ:
                    del os.environ['CUDA_LAUNCH_BLOCKING']

                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True

            except Exception as finally_error:
                logger.warning(f"Finally block cleanup failed: {finally_error}")

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
