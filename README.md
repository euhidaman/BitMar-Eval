# BitMar Model Evaluation Suite

This repository contains a comprehensive evaluation suite for the BitMar model, designed to run 7 key language model benchmarks that are most compatible with BitMar's training data.

## Overview

The evaluation suite automatically:
- Downloads 7 selected benchmark datasets optimized for BitMar's training data
- Loads your trained BitMar model from Hugging Face Hub
- Runs evaluations on benchmarks that align with your training sources
- Generates detailed results and summaries
- Supports both GPU and CPU evaluation

## Model Information

- **Model**: `euhidaman/bitmar-attention-multimodal`
- **Training**: 100M tokens with perfect image-caption alignment
- **Architecture**: BitNet-based multimodal transformer with episodic memory

## Training Data Compatibility

Your BitMar model was trained on the `train_50M` dataset containing:
- `bnc_spoken.train` - British National Corpus spoken language
- `childes.train` - Child language data  
- `gutenberg.train` - Literature texts
- `open_subtitles.train` - Subtitle text
- `simple_wiki.train` - Simplified Wikipedia
- `switchboard.train` - Conversational speech transcripts

The selected benchmarks test language understanding that should align well with this training data.

## Requirements

### System Requirements
- **GPU**: NVIDIA A100 (recommended) or other CUDA-compatible GPU
- **Memory**: 16+ GB RAM, 8+ GB VRAM
- **Storage**: ~8 GB for selected datasets (reduced from 15 GB)
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
2. Download 7 selected benchmark datasets (~8 GB)
3. Test model loading
4. Run 7 optimized benchmarks
5. Generate comprehensive results

**Expected Runtime**: 2-3 hours total (reduced from 3-5 hours)

### Option 2: Manual Step-by-Step Setup

If you prefer manual control or the automated script fails:

```bash
# 1. Navigate to directory
cd /path/to/BitMar-Eval

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test model loading (optional but recommended)
python test_model_loading.py

# 4. Download selected benchmark datasets
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

## Benchmark Details

### Selected Benchmarks (Optimized for BitMar Training Data)

| Benchmark | Type | Shots | Metric | Description | Training Data Alignment |
|-----------|------|-------|---------|-------------|-------------------------|
| **MMLU** | 5-shot | 5 | Accuracy | Massive multitask understanding | ✅ Wikipedia content |
| **HellaSwag** | 0-shot | 0 | Accuracy | Commonsense reasoning | ✅ Conversational/subtitle data |
| **WinoGrande** | 0-shot | 0 | Accuracy | Pronoun resolution | ✅ Diverse text sources |
| **BoolQ** | 0-shot | 0 | Accuracy | Reading comprehension | ✅ Wikipedia + diverse text |
| **ARC-Easy** | 0-shot | 0 | Accuracy | Basic science reasoning | ✅ Simplified content |
| **CommonsenseQA** | 10-shot | 10 | Accuracy | Commonsense knowledge | ✅ Conversational understanding |
| **PIQA** | 0-shot | 0 | Accuracy | Physical reasoning | ✅ General language understanding |

### Expected Runtimes

On NVIDIA A100:
- **Dataset Download**: 20-40 minutes (reduced)
- **ARC-Easy**: ~10 minutes
- **BoolQ**: ~8 minutes
- **HellaSwag**: ~12 minutes
- **PIQA**: ~5 minutes
- **WinoGrande**: ~8 minutes
- **CommonsenseQA**: ~20 minutes
- **MMLU**: ~45 minutes

**Total Evaluation Time**: 1.5-2 hours (reduced from 2-4 hours)

## Why These 7 Benchmarks?

These benchmarks were selected because they:

1. **Align with Training Data**: Test language understanding skills present in your training sources
2. **Avoid Domain Mismatch**: Skip highly specialized benchmarks that require knowledge not in your training data
3. **Cover Core Skills**: Test reading comprehension, commonsense reasoning, and basic knowledge
4. **Efficient Evaluation**: Faster runtime while maintaining comprehensive assessment

**Excluded Benchmarks** (and why):
- ARC-Challenge: Too advanced for training data scope
- OpenbookQA: Specialized science knowledge
- TruthfulQA: Tests truthfulness (challenging for diverse training sources)
- TriviaQA: Factual knowledge beyond training scope

## Results and Output

### Output Structure
```
evaluation_results/
├── bitmar_evaluation_20250814_143022.json    # Detailed results
├── bitmar_summary_20250814_143022.txt        # Human-readable summary
└── ...

dataset_cache/
├── ai2_arc/                                  # Only ARC-Easy
├── hellaswag/
├── super_glue/                              # BoolQ
├── piqa/
├── winogrande/
├── commonsense_qa/
├── cais--mmlu/                              # All MMLU subjects
└── ...
```

### Results Format

#### Summary File Example
```
BitMar Model Evaluation Summary
========================================

Model: euhidaman/bitmar-attention-multimodal
Date: 2025-08-14T14:30:22
Total Time: 5247.32 seconds
Training Data Optimized: ✅

ARC-Easy:
  ✅ Accuracy: 0.4321

BoolQ:
  ✅ Accuracy: 0.6124

HellaSwag:
  ✅ Accuracy: 0.3456

PIQA:
  ✅ Accuracy: 0.5234

WinoGrande:
  ✅ Accuracy: 0.5123

CommonsenseQA:
  ✅ Accuracy: 0.4567

MMLU:
  ✅ Overall Accuracy: 0.3234
```

## Commands for A100 Server

```bash
# Complete pipeline in one go:
cd /path/to/BitMar-Eval
pip install -r requirements.txt
python test_model_loading.py
python download_datasets.py --cache_dir ./dataset_cache
python evaluate_bitmar_benchmarks.py \
    --model_path "euhidaman/bitmar-attention-multimodal" \
    --device "cuda" \
    --batch_size 16 \
    --output_dir evaluation_results
```

## Performance Expectations

Given your BitMar model's training on 100M tokens from diverse text sources, you can expect:

- **Strong Performance**: BoolQ, HellaSwag (conversational reasoning)
- **Good Performance**: WinoGrande, PIQA (general language understanding)
- **Moderate Performance**: ARC-Easy, CommonsenseQA (basic reasoning)
- **Variable Performance**: MMLU (depends on subject overlap with training data)

The evaluation focuses on testing capabilities that your model should have learned from its training data, providing meaningful insights into its language understanding abilities.

## Storage Requirements

- **Datasets**: ~8 GB (reduced from 15 GB)
- **Model Cache**: ~2 GB
- **Results**: ~100 MB
- **Total**: ~10 GB

## Citation

If you use this evaluation suite in your research, please cite:

```bibtex
@misc{bitmar_eval_suite,
  title={BitMar Evaluation Suite - Training Data Optimized},
  author={BitMar Team},
  year={2025},
  howpublished={\url{https://github.com/your-repo/bitmar-eval}}
}
```
