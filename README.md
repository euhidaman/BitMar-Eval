# BitMar Model Evaluation Suite

This repository contains a comprehensive evaluation suite for the BitMar model, designed to run all major language model benchmarks including ARC, OpenbookQA, BoolQ, HellaSwag, PIQA, WinoGrande, CommonsenseQA, TruthfulQA, TriviaQA, and MMLU.

## Overview

The evaluation suite automatically:
- Downloads all required benchmark datasets
- Loads your trained BitMar model from Hugging Face Hub
- Runs comprehensive evaluations across 11 different benchmarks
- Generates detailed results and summaries
- Supports both GPU and CPU evaluation

## Model Information

- **Model**: `euhidaman/bitmar-attention-multimodal`
- **Training**: 100M tokens with perfect image-caption alignment
- **Architecture**: BitNet-based multimodal transformer with episodic memory

## Requirements

### System Requirements
- **GPU**: NVIDIA A100 (recommended) or other CUDA-compatible GPU
- **Memory**: 16+ GB RAM, 8+ GB VRAM
- **Storage**: ~15 GB for datasets
- **Python**: 3.8+

### Software Dependencies
All dependencies are listed in `requirements.txt` and will be installed automatically.

## Quick Start

### Option 1: Automated Setup and Evaluation (Recommended)

```bash
# Navigate to BitMar-Eval directory
cd /path/to/BitMar-Eval

# Run the complete automated pipeline
./run_evaluation.sh
```

This single command will:
1. Install all dependencies
2. Download all benchmark datasets (~15 GB)
3. Test model loading
4. Run all 11 benchmarks
5. Generate comprehensive results

**Expected Runtime**: 3-5 hours total (including dataset download)

### Option 2: Manual Step-by-Step Setup

If you prefer manual control or the automated script fails:

```bash
# 1. Navigate to directory
cd /path/to/BitMar-Eval

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test model loading (optional but recommended)
python test_model_loading.py

# 4. Download benchmark datasets
python download_datasets.py --cache_dir ./dataset_cache

# 5. Verify dataset integrity
python download_datasets.py --cache_dir ./dataset_cache --verify_only

# 6. Run evaluation
python evaluate_bitmar_benchmarks.py \
    --model_path "euhidaman/bitmar-attention-multimodal" \
    --device "cuda" \
    --batch_size 16 \
    --output_dir evaluation_results
```

## Detailed Usage Instructions

### 1. Environment Setup

#### Python Environment (Recommended)
```bash
# Create virtual environment
python -m venv bitmar_eval_env
source bitmar_eval_env/bin/activate  # Linux/Mac
# or
bitmar_eval_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

#### Required Packages
The evaluation suite requires:
- PyTorch 2.0+
- Transformers 4.30+
- Datasets 2.12+
- Accelerate 0.20+
- And other packages listed in `requirements.txt`

### 2. Model Testing

Before running the full evaluation, test if your model loads correctly:

```bash
python test_model_loading.py
```

**Expected Output**:
```
✅ CUDA available: NVIDIA A100-SXM4-80GB
✅ Model loaded successfully!
✅ Tokenization test passed: torch.Size([1, 6])
✅ Generation test passed: 'Paris'
🎉 All tests passed! BitMar model is ready for evaluation.
```

### 3. Dataset Download

Download all required benchmark datasets:

```bash
# Download all datasets (30-60 minutes)
python download_datasets.py --cache_dir ./dataset_cache

# Verify download success
python download_datasets.py --cache_dir ./dataset_cache --verify_only
```

**Datasets Downloaded**:
- ARC-Challenge & ARC-Easy
- OpenbookQA
- BoolQ (SuperGLUE)
- HellaSwag
- PIQA
- WinoGrande
- CommonsenseQA
- TruthfulQA
- TriviaQA
- MMLU (all 57 subjects)

**Storage Requirements**: ~15 GB total

### 4. Running Evaluations

#### Full Evaluation (All Benchmarks)
```bash
python evaluate_bitmar_benchmarks.py \
    --model_path "euhidaman/bitmar-attention-multimodal" \
    --device "cuda" \
    --batch_size 16 \
    --output_dir evaluation_results
```

#### Custom Evaluation Options

**Different Batch Sizes**:
```bash
# For limited GPU memory
python evaluate_bitmar_benchmarks.py \
    --model_path "euhidaman/bitmar-attention-multimodal" \
    --device "cuda" \
    --batch_size 4 \
    --output_dir evaluation_results

# For high-memory GPUs
python evaluate_bitmar_benchmarks.py \
    --model_path "euhidaman/bitmar-attention-multimodal" \
    --device "cuda" \
    --batch_size 32 \
    --output_dir evaluation_results
```

**CPU-Only Evaluation**:
```bash
python evaluate_bitmar_benchmarks.py \
    --model_path "euhidaman/bitmar-attention-multimodal" \
    --device "cpu" \
    --batch_size 2 \
    --output_dir evaluation_results
```

**Note**: CPU evaluation will be significantly slower (10-15x) but works if GPU is unavailable.

### 5. Monitoring Progress

#### Real-time Log Monitoring
```bash
# Monitor evaluation progress
tail -f bitmar_evaluation.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

#### Progress Indicators
The evaluation will show progress for each benchmark:
```
🧠 Evaluating ARC-Challenge (0-shot)
ARC-Challenge progress: 50 samples processed
ARC-Challenge progress: 100 samples processed
...
✅ ARC-Challenge completed: 0.2845 accuracy (342/1172)
```

## Benchmark Details

### Evaluation Configurations

| Benchmark | Type | Shots | Metric | Description |
|-----------|------|-------|---------|-------------|
| ARC-Challenge | 0-shot | 0 | Accuracy | AI2 Reasoning Challenge (hard) |
| ARC-Easy | 0-shot | 0 | Accuracy | AI2 Reasoning Challenge (easy) |
| OpenbookQA | 0-shot | 0 | Accuracy | Open-book question answering |
| BoolQ | 0-shot | 0 | Accuracy | Boolean questions |
| HellaSwag | 0-shot | 0 | Accuracy | Commonsense reasoning |
| PIQA | 0-shot | 0 | Accuracy | Physical interaction QA |
| WinoGrande | 0-shot | 0 | Accuracy | Winograd schema challenge |
| CommonsenseQA | 10-shot | 10 | Accuracy | Commonsense reasoning |
| TruthfulQA | 10-shot | 10 | MC2 | Truthfulness evaluation |
| TriviaQA | 5-shot | 5 | EM | Exact match on trivia |
| MMLU | 5-shot | 5 | Accuracy | Massive multitask understanding |

### Expected Runtimes

On NVIDIA A100:
- **Dataset Download**: 30-60 minutes
- **ARC-Challenge**: ~15 minutes
- **ARC-Easy**: ~10 minutes
- **OpenbookQA**: ~5 minutes
- **BoolQ**: ~8 minutes
- **HellaSwag**: ~12 minutes
- **PIQA**: ~5 minutes
- **WinoGrande**: ~8 minutes
- **CommonsenseQA**: ~20 minutes
- **TruthfulQA**: ~15 minutes
- **TriviaQA**: ~25 minutes
- **MMLU**: ~45 minutes

**Total Evaluation Time**: 2-4 hours

## Results and Output

### Output Structure
```
evaluation_results/
├── bitmar_evaluation_20250814_143022.json    # Detailed results
├── bitmar_summary_20250814_143022.txt        # Human-readable summary
└── ...

logs/
├── bitmar_evaluation_20250814_143022.log     # Execution log
└── ...

dataset_cache/
├── ai2_arc/                                  # Cached datasets
├── hellaswag/
├── ...
```

### Results Format

#### Summary File Example
```
BitMar Model Evaluation Summary
========================================

Model: euhidaman/bitmar-attention-multimodal
Date: 2025-08-14T14:30:22
Total Time: 8247.32 seconds

ARC-Challenge:
  ✅ Accuracy: 0.2845

ARC-Easy:
  ✅ Accuracy: 0.4321

BoolQ:
  ✅ Accuracy: 0.6124

...
```

#### JSON Results Structure
```json
{
  "model_path": "euhidaman/bitmar-attention-multimodal",
  "evaluation_date": "2025-08-14T14:30:22",
  "total_evaluation_time": 8247.32,
  "results": {
    "ARC-Challenge": {
      "accuracy": 0.2845,
      "correct": 342,
      "total": 1172
    },
    ...
  }
}
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Reduce batch size
python evaluate_bitmar_benchmarks.py ... --batch_size 4

# Use CPU instead
python evaluate_bitmar_benchmarks.py ... --device cpu
```

#### 2. Model Loading Fails
**Error**: `OSError: Repository not found`

**Solutions**:
- Ensure you have internet connection
- Check if model repository exists: https://huggingface.co/euhidaman/bitmar-attention-multimodal
- Try authentication if repository is private

#### 3. Dataset Download Fails
**Error**: `Connection timeout` or `Dataset not found`

**Solutions**:
```bash
# Retry download
python download_datasets.py --cache_dir ./dataset_cache --force_download

# Check internet connection
# Try with smaller timeout or different mirror
```

#### 4. Evaluation Hangs
**Symptoms**: No progress for >30 minutes

**Solutions**:
```bash
# Check GPU memory
nvidia-smi

# Restart with smaller batch size
# Check logs for specific errors
tail -f bitmar_evaluation.log
```

### Performance Optimization

#### For Faster Evaluation
```bash
# Use larger batch size if memory allows
--batch_size 32

# Use multiple workers (if dataset supports)
# Set optimal number of workers for your system
```

#### For Limited Resources
```bash
# Minimal memory usage
--batch_size 1 --device cpu

# Reduce evaluation scope (modify script to run specific benchmarks)
```

### Getting Help

1. **Check Logs**: Always check `bitmar_evaluation.log` for detailed error messages
2. **Monitor Resources**: Use `nvidia-smi` and `htop` to monitor GPU/CPU usage
3. **Verify Setup**: Run `test_model_loading.py` to ensure basic functionality

## Advanced Usage

### Custom Benchmark Selection

To run specific benchmarks, modify the `evaluate_bitmar_benchmarks.py` script or create a custom evaluation script:

```python
# Example: Run only ARC benchmarks
evaluator = BitMarEvaluator(model_path="euhidaman/bitmar-attention-multimodal")
arc_challenge_results = evaluator.evaluate_arc_challenge()
arc_easy_results = evaluator.evaluate_arc_easy()
```

### Integration with Other Pipelines

The evaluation results can be integrated with other evaluation pipelines:

```bash
# Save results in specific format for downstream analysis
python evaluate_bitmar_benchmarks.py \
    --output_dir evaluation_results \
    --save_format json,csv,txt
```

### Batch Processing Multiple Models

To evaluate multiple model checkpoints:

```bash
# Create a script to iterate through checkpoints
for checkpoint in checkpoints/*; do
    python evaluate_bitmar_benchmarks.py \
        --model_path "$checkpoint" \
        --output_dir "results_$(basename $checkpoint)"
done
```

## Citation

If you use this evaluation suite in your research, please cite:

```bibtex
@misc{bitmar_eval_suite,
  title={BitMar Comprehensive Evaluation Suite},
  author={BitMar Team},
  year={2025},
  howpublished={\url{https://github.com/your-repo/bitmar-eval}}
}
```

## License

This evaluation suite is released under the MIT License. See LICENSE file for details.

## Acknowledgments

This evaluation suite uses datasets and evaluation frameworks from:
- AI2 (ARC, OpenbookQA)
- SuperGLUE (BoolQ)
- Various academic institutions (HellaSwag, PIQA, WinoGrande, etc.)
- Hugging Face Datasets library

Special thanks to the BabyLM challenge organizers and the broader NLP community.
