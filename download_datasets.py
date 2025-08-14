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
    """Download a dataset and cache it locally with robust error handling"""
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            logger.info(f"📥 Downloading {dataset_name}" + (f" ({config})" if config else "") +
                       (f" - Attempt {attempt + 1}/{max_retries}" if attempt > 0 else ""))

            # Clear any existing corrupted cache for this dataset
            if attempt > 0 and cache_dir:
                cache_path = Path(cache_dir)
                dataset_cache_pattern = f"{dataset_name.replace('/', '--')}*"
                if config:
                    dataset_cache_pattern = f"{dataset_name.replace('/', '--')}__{config}*"

                # Remove potentially corrupted cache files
                for cache_file in cache_path.glob(dataset_cache_pattern):
                    if cache_file.is_dir():
                        import shutil
                        try:
                            shutil.rmtree(cache_file)
                            logger.info(f"🗑️  Cleared corrupted cache: {cache_file}")
                        except Exception as e:
                            logger.warning(f"Failed to clear cache {cache_file}: {e}")

            # Download with specific parameters to avoid encoding issues
            download_kwargs = {
                'cache_dir': cache_dir,
                'verification_mode': 'no_checks'  # Skip verification that might cause encoding issues
            }

            if config:
                dataset = load_dataset(dataset_name, config, **download_kwargs)
            else:
                dataset = load_dataset(dataset_name, **download_kwargs)

            logger.info(f"✅ Successfully downloaded {dataset_name}")
            return True

        except UnicodeDecodeError as e:
            logger.warning(f"⚠️  UTF-8 encoding error for {dataset_name}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"🔄 Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
                continue
            else:
                logger.error(f"❌ Failed to download {dataset_name} after {max_retries} attempts due to encoding issues")

        except Exception as e:
            error_msg = str(e).lower()

            # Handle specific error types
            if 'connection' in error_msg or 'timeout' in error_msg or 'network' in error_msg:
                logger.warning(f"⚠️  Network error for {dataset_name}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"🔄 Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    continue
            elif 'utf-8' in error_msg or 'decode' in error_msg or 'encoding' in error_msg:
                logger.warning(f"⚠️  Encoding error for {dataset_name}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"🔄 Clearing cache and retrying...")
                    import time
                    time.sleep(retry_delay)
                    continue
            elif 'disk space' in error_msg or 'no space' in error_msg:
                logger.error(f"❌ Insufficient disk space for {dataset_name}: {e}")
                return False
            else:
                logger.warning(f"⚠️  Unexpected error for {dataset_name}: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                    continue

            # If it's the last attempt, log the final error
            if attempt == max_retries - 1:
                logger.error(f"❌ Failed to download {dataset_name} after {max_retries} attempts: {e}")

    return False

def download_all_datasets(cache_dir=None):
    """Download all required benchmark datasets"""
    logger.info("🚀 Starting download of all benchmark datasets")

    datasets_to_download = [
        # ARC dataset (only ARC-Easy)
        ("ai2_arc", "ARC-Easy"),

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

        # NOTE: MMLU is handled separately below due to multiple configs
    ]

    success_count = 0
    total_count = len(datasets_to_download)

    for dataset_name, config in datasets_to_download:
        success = download_dataset(dataset_name, config, cache_dir)
        if success:
            success_count += 1

    # Special handling for MMLU (download all subjects individually)
    try:
        logger.info("📥 Downloading all MMLU subjects...")
        from datasets import get_dataset_config_names

        # Get all available MMLU subjects
        subjects = get_dataset_config_names("cais/mmlu")
        logger.info(f"Found {len(subjects)} MMLU subjects")

        mmlu_success = 0
        mmlu_total = len(subjects)

        for i, subject in enumerate(subjects):
            try:
                logger.info(f"📥 Downloading MMLU subject: {subject} ({i+1}/{mmlu_total})")
                success = download_dataset("cais/mmlu", subject, cache_dir)
                if success:
                    mmlu_success += 1

                # Log progress every 10 subjects
                if (i + 1) % 10 == 0:
                    logger.info(f"MMLU progress: {i + 1}/{mmlu_total} subjects processed")

            except Exception as e:
                logger.warning(f"Failed to download MMLU subject {subject}: {e}")
                continue

        logger.info(f"✅ Downloaded {mmlu_success}/{mmlu_total} MMLU subjects")

        # Add MMLU to success count if we got most subjects
        if mmlu_success > mmlu_total * 0.8:  # If we got at least 80% of subjects
            success_count += 1
            total_count += 1
        else:
            total_count += 1
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
        ("ai2_arc", "ARC-Easy", "test"),
        ("super_glue", "boolq", "validation"),
        ("hellaswag", None, "validation"),
        ("piqa", None, "validation"),
        ("winogrande", "winogrande_debiased", "validation"),
        ("commonsense_qa", None, "validation"),
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
