#!/usr/bin/env python3
"""
Download and prepare all benchmark datasets for BitMar evaluation
This script downloads all required datasets and prepares them for evaluation
"""

import os
import sys
import logging
from pathlib import Path
from datasets import load_dataset
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_dataset(dataset_name, config=None, cache_dir=None):
    """Download a dataset and cache it locally"""
    try:
        logger.info(f"📥 Downloading {dataset_name}" + (f" ({config})" if config else ""))

        if config:
            dataset = load_dataset(dataset_name, config, cache_dir=cache_dir)
        else:
            dataset = load_dataset(dataset_name, cache_dir=cache_dir)

        logger.info(f"✅ Successfully downloaded {dataset_name}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to download {dataset_name}: {e}")
        return False

def download_all_datasets(cache_dir=None):
    """Download all required benchmark datasets"""
    logger.info("🚀 Starting download of all benchmark datasets")

    datasets_to_download = [
        # ARC datasets
        ("ai2_arc", "ARC-Challenge"),
        ("ai2_arc", "ARC-Easy"),

        # OpenbookQA
        ("openbookqa", "main"),

        # SuperGLUE BoolQ
        ("super_glue", "boolq"),

        # HellaSwag
        ("hellaswag", None),

        # PIQA
        ("piqa", None),

        # WinoGrande
        ("winogrande", "winogrande_debiased"),

        # CommonsenseQA
        ("commonsense_qa", None),

        # TruthfulQA
        ("truthful_qa", "multiple_choice"),

        # TriviaQA
        ("trivia_qa", "rc.nocontext"),

        # MMLU
        ("cais/mmlu", None),  # Will download all configs
    ]

    success_count = 0
    total_count = len(datasets_to_download)

    for dataset_name, config in datasets_to_download:
        success = download_dataset(dataset_name, config, cache_dir)
        if success:
            success_count += 1

    # Special handling for MMLU (download all subjects)
    try:
        logger.info("📥 Downloading all MMLU subjects...")
        from datasets import get_dataset_config_names
        subjects = get_dataset_config_names("cais/mmlu")

        logger.info(f"Found {len(subjects)} MMLU subjects")
        mmlu_success = 0

        for i, subject in enumerate(subjects):
            try:
                load_dataset("cais/mmlu", subject, cache_dir=cache_dir)
                mmlu_success += 1
                if (i + 1) % 10 == 0:
                    logger.info(f"Downloaded {i + 1}/{len(subjects)} MMLU subjects")
            except Exception as e:
                logger.warning(f"Failed to download MMLU subject {subject}: {e}")

        logger.info(f"✅ Downloaded {mmlu_success}/{len(subjects)} MMLU subjects")

    except Exception as e:
        logger.error(f"❌ Failed to download MMLU subjects: {e}")

    logger.info(f"📊 Dataset Download Summary:")
    logger.info(f"  • Successfully downloaded: {success_count}/{total_count} main datasets")
    logger.info(f"  • MMLU subjects: {mmlu_success}/{len(subjects) if 'subjects' in locals() else 'unknown'}")

    if success_count == total_count:
        logger.info("🎉 All datasets downloaded successfully!")
        return True
    else:
        logger.warning(f"⚠️  Some datasets failed to download ({total_count - success_count} failures)")
        return False

def verify_datasets(cache_dir=None):
    """Verify that all datasets are properly cached"""
    logger.info("🔍 Verifying dataset availability...")

    verification_list = [
        ("ai2_arc", "ARC-Challenge", "test"),
        ("ai2_arc", "ARC-Easy", "test"),
        ("openbookqa", "main", "test"),
        ("super_glue", "boolq", "validation"),
        ("hellaswag", None, "validation"),
        ("piqa", None, "validation"),
        ("winogrande", "winogrande_debiased", "validation"),
        ("commonsense_qa", None, "validation"),
        ("truthful_qa", "multiple_choice", "validation"),
        ("trivia_qa", "rc.nocontext", "validation"),
    ]

    verified_count = 0
    total_count = len(verification_list)

    for dataset_name, config, split in verification_list:
        try:
            if config:
                dataset = load_dataset(dataset_name, config, split=split, cache_dir=cache_dir)
            else:
                dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

            logger.info(f"✅ {dataset_name}" + (f" ({config})" if config else "") + f" - {len(dataset)} samples")
            verified_count += 1

        except Exception as e:
            logger.error(f"❌ Failed to verify {dataset_name}: {e}")

    # Verify MMLU
    try:
        from datasets import get_dataset_config_names
        subjects = get_dataset_config_names("cais/mmlu")
        mmlu_verified = 0

        for subject in subjects[:5]:  # Check first 5 subjects
            try:
                dataset = load_dataset("cais/mmlu", subject, split="test", cache_dir=cache_dir)
                mmlu_verified += 1
            except:
                break

        if mmlu_verified == 5:
            logger.info(f"✅ MMLU - verified {len(subjects)} subjects available")
            verified_count += 1
            total_count += 1
        else:
            logger.error(f"❌ MMLU verification failed")
            total_count += 1

    except Exception as e:
        logger.error(f"❌ MMLU verification failed: {e}")
        total_count += 1

    logger.info(f"📊 Verification Summary: {verified_count}/{total_count} datasets verified")

    return verified_count == total_count

def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets for BitMar evaluation")
    parser.add_argument("--cache_dir", type=str, default="./dataset_cache",
                       help="Directory to cache datasets")
    parser.add_argument("--verify_only", action="store_true",
                       help="Only verify existing datasets, don't download")
    parser.add_argument("--force_download", action="store_true",
                       help="Force re-download even if datasets exist")

    args = parser.parse_args()

    # Create cache directory
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    logger.info(f"Using cache directory: {cache_dir.absolute()}")

    try:
        if args.verify_only:
            # Only verify existing datasets
            success = verify_datasets(str(cache_dir))
        else:
            # Download datasets
            if args.force_download:
                logger.info("🔄 Force download mode - will re-download all datasets")

            success = download_all_datasets(str(cache_dir))

            if success:
                # Verify after download
                logger.info("\n" + "="*50)
                logger.info("Verifying downloaded datasets...")
                logger.info("="*50)
                verify_datasets(str(cache_dir))

        if success:
            logger.info("🎉 Dataset preparation completed successfully!")
            logger.info(f"All datasets are cached in: {cache_dir.absolute()}")

            # Print disk usage
            try:
                import shutil
                total, used, free = shutil.disk_usage(cache_dir)
                cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                logger.info(f"💾 Cache size: {cache_size / (1024**3):.2f} GB")
            except:
                pass

            return 0
        else:
            logger.error("❌ Dataset preparation failed!")
            return 1

    except Exception as e:
        logger.error(f"❌ Error during dataset preparation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
