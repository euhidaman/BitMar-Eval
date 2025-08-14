#!/bin/bash
# BitMar Evaluation Setup Script for A100 Server
# This script sets up the environment and runs comprehensive evaluations

set -e  # Exit on any error

echo "🚀 BitMar Evaluation Setup for A100 Server"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "evaluate_bitmar_benchmarks.py" ]; then
    echo "❌ Error: Please run this script from the BitMar-Eval directory"
    exit 1
fi

# Function to print status
print_status() {
    echo ""
    echo "📍 $1"
    echo "----------------------------------------"
}

# Create necessary directories
print_status "Creating directories"
mkdir -p dataset_cache
mkdir -p evaluation_results
mkdir -p logs

# Check GPU availability
print_status "Checking GPU availability"
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA drivers found"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "⚠️  NVIDIA drivers not found - will use CPU"
fi

# Check Python version
print_status "Checking Python environment"
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python version: $python_version"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✅ Python version is compatible"
else
    echo "❌ Python 3.8+ required"
    exit 1
fi

# Install requirements
print_status "Installing Python requirements"
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
    echo "✅ Requirements installed"
else
    echo "❌ requirements.txt not found"
    exit 1
fi

# Download datasets
print_status "Downloading benchmark datasets (this may take a while)"
echo "📥 Starting dataset download..."
python3 download_datasets.py --cache_dir ./dataset_cache

if [ $? -eq 0 ]; then
    echo "✅ All datasets downloaded successfully"
else
    echo "⚠️  Some datasets may have failed to download, but continuing..."
fi

# Verify datasets
print_status "Verifying dataset integrity"
python3 download_datasets.py --cache_dir ./dataset_cache --verify_only

# Run the evaluation
print_status "Starting BitMar comprehensive evaluation"
echo "🧠 This will run all benchmarks: ARC, OpenbookQA, BoolQ, HellaSwag, PIQA, WinoGrande, CommonsenseQA, TruthfulQA, TriviaQA, MMLU"
echo ""

# Set environment variables for optimal GPU usage
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run evaluation with logging
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/bitmar_evaluation_${timestamp}.log"

echo "📊 Starting evaluation (logging to $log_file)"
echo "⏱️  Expected time: 2-4 hours depending on GPU"

python3 evaluate_bitmar_benchmarks.py \
    --model_path "euhidaman/bitmar-attention-multimodal" \
    --device "cuda" \
    --batch_size 8 \
    --output_dir evaluation_results 2>&1 | tee "$log_file"

if [ $? -eq 0 ]; then
    print_status "Evaluation completed successfully!"
    echo "📊 Results saved in: evaluation_results/"
    echo "📝 Log saved in: $log_file"

    # Show summary if available
    latest_summary=$(ls -t evaluation_results/bitmar_summary_*.txt 2>/dev/null | head -n1)
    if [ -f "$latest_summary" ]; then
        echo ""
        echo "📋 Evaluation Summary:"
        echo "====================="
        cat "$latest_summary"
    fi
else
    echo "❌ Evaluation failed. Check the log file: $log_file"
    exit 1
fi

echo ""
echo "🎉 BitMar evaluation pipeline completed!"
echo "Results are available in the evaluation_results/ directory"
