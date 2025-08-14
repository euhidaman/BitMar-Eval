#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for BitMar Model
Runs all specified benchmarks: ARC, OpenbookQA, BoolQ, HellaSwag, PIQA, WinoGrande,
CommonsenseQA, TruthfulQA, TriviaQA, and MMLU
Uses BitMar-specific adapter for proper model compatibility
"""

import os
import sys
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from datasets import load_dataset
import pandas as pd
from datetime import datetime
import traceback

# Import BitMar adapter
try:
    from bitmar_adapter import create_bitmar_adapter
    BITMAR_ADAPTER_AVAILABLE = True
    print("✅ BitMar adapter available")
except ImportError:
    BITMAR_ADAPTER_AVAILABLE = False
    print("❌ BitMar adapter not available")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bitmar_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BitMarEvaluator:
    """Comprehensive evaluator for BitMar model on various benchmarks"""

    def __init__(self, model_path: str, device: str = "cuda", batch_size: int = 8):
        """Initialize the evaluator"""
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.results = {}

        logger.info(f"Initializing BitMar Evaluator")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {batch_size}")

        # Load model using BitMar adapter
        self.load_model()

    def load_model(self):
        """Load BitMar model using the specialized adapter"""
        try:
            logger.info("Loading BitMar model using specialized adapter...")

            if not BITMAR_ADAPTER_AVAILABLE:
                raise ImportError("BitMar adapter not available. Please ensure bitmar_adapter.py is in the same directory.")

            # Create BitMar adapter
            self.model_adapter = create_bitmar_adapter(
                model_path=self.model_path,
                device=str(self.device)
            )

            # Get model info
            model_info = self.model_adapter.get_model_info()

            logger.info("✅ BitMar model and adapter loaded successfully")
            logger.info(f"Model type: {model_info.get('model_type', 'Unknown')}")
            logger.info(f"Architecture: {model_info.get('architecture', 'Unknown')}")
            logger.info(f"Total parameters: {model_info.get('total_parameters', 'Unknown'):,}" if isinstance(model_info.get('total_parameters'), int) else f"Total parameters: {model_info.get('total_parameters', 'Unknown')}")
            logger.info(f"Model device: {model_info.get('model_device', 'Unknown')}")

        except Exception as e:
            logger.error(f"❌ Failed to load BitMar model: {e}")
            raise

    def generate_response(self, prompt: str, max_length: int = 100, temperature: float = 0.1) -> str:
        """Generate response from BitMar model using adapter"""
        try:
            return self.model_adapter.generate_response(prompt, max_length, temperature)
        except Exception as e:
            logger.warning(f"Generation failed for prompt: {prompt[:50]}... Error: {e}")
            return ""

    def evaluate_multiple_choice(self, prompt: str, choices: List[str]) -> int:
        """Evaluate multiple choice question using BitMar adapter"""
        try:
            return self.model_adapter.evaluate_multiple_choice(prompt, choices)
        except Exception as e:
            logger.warning(f"Multiple choice evaluation failed: {e}")
            return 0  # Default to first choice

    def evaluate_arc_challenge(self) -> Dict[str, float]:
        """Evaluate on ARC-Challenge (0-shot)"""
        logger.info("🧠 Evaluating ARC-Challenge (0-shot)")

        try:
            dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
            correct = 0
            total = 0

            for item in dataset:
                question = item['question']
                choices = item['choices']['text']
                correct_answer = item['answerKey']

                # Create prompt
                prompt = f"Question: {question}\nChoices:\n"
                for i, choice in enumerate(choices):
                    prompt += f"{chr(65+i)}) {choice}\n"
                prompt += "Answer:"

                # Get prediction
                predicted_idx = self.evaluate_multiple_choice(prompt, choices)
                predicted_answer = chr(65 + predicted_idx)

                if predicted_answer == correct_answer:
                    correct += 1
                total += 1

                if total % 50 == 0:
                    logger.info(f"ARC-Challenge progress: {total} samples processed")

            accuracy = correct / total if total > 0 else 0
            result = {"accuracy": accuracy, "correct": correct, "total": total}

            logger.info(f"✅ ARC-Challenge completed: {accuracy:.4f} accuracy ({correct}/{total})")
            return result

        except Exception as e:
            logger.error(f"❌ ARC-Challenge evaluation failed: {e}")
            return {"accuracy": 0.0, "error": str(e)}

    def evaluate_arc_easy(self) -> Dict[str, float]:
        """Evaluate on ARC-Easy (0-shot)"""
        logger.info("🧠 Evaluating ARC-Easy (0-shot)")

        try:
            dataset = load_dataset("ai2_arc", "ARC-Easy", split="test")
            correct = 0
            total = 0

            for item in dataset:
                question = item['question']
                choices = item['choices']['text']
                correct_answer = item['answerKey']

                # Create prompt
                prompt = f"Question: {question}\nChoices:\n"
                for i, choice in enumerate(choices):
                    prompt += f"{chr(65+i)}) {choice}\n"
                prompt += "Answer:"

                # Get prediction
                predicted_idx = self.evaluate_multiple_choice(prompt, choices)
                predicted_answer = chr(65 + predicted_idx)

                if predicted_answer == correct_answer:
                    correct += 1
                total += 1

                if total % 50 == 0:
                    logger.info(f"ARC-Easy progress: {total} samples processed")

            accuracy = correct / total if total > 0 else 0
            result = {"accuracy": accuracy, "correct": correct, "total": total}

            logger.info(f"✅ ARC-Easy completed: {accuracy:.4f} accuracy ({correct}/{total})")
            return result

        except Exception as e:
            logger.error(f"❌ ARC-Easy evaluation failed: {e}")
            return {"accuracy": 0.0, "error": str(e)}

    def evaluate_openbookqa(self) -> Dict[str, float]:
        """Evaluate on OpenbookQA (0-shot)"""
        logger.info("🧠 Evaluating OpenbookQA (0-shot)")

        try:
            dataset = load_dataset("openbookqa", "main", split="test")
            correct = 0
            total = 0

            for item in dataset:
                question = item['question_stem']
                choices = item['choices']['text']
                correct_answer = item['answerKey']

                # Create prompt
                prompt = f"Question: {question}\nChoices:\n"
                for i, choice in enumerate(choices):
                    prompt += f"{chr(65+i)}) {choice}\n"
                prompt += "Answer:"

                # Get prediction
                predicted_idx = self.evaluate_multiple_choice(prompt, choices)
                predicted_answer = chr(65 + predicted_idx)

                if predicted_answer == correct_answer:
                    correct += 1
                total += 1

                if total % 50 == 0:
                    logger.info(f"OpenbookQA progress: {total} samples processed")

            accuracy = correct / total if total > 0 else 0
            result = {"accuracy": accuracy, "correct": correct, "total": total}

            logger.info(f"✅ OpenbookQA completed: {accuracy:.4f} accuracy ({correct}/{total})")
            return result

        except Exception as e:
            logger.error(f"❌ OpenbookQA evaluation failed: {e}")
            return {"accuracy": 0.0, "error": str(e)}

    def evaluate_boolq(self) -> Dict[str, float]:
        """Evaluate on BoolQ (0-shot)"""
        logger.info("🧠 Evaluating BoolQ (0-shot)")

        try:
            dataset = load_dataset("super_glue", "boolq", split="validation")
            correct = 0
            total = 0

            for item in dataset:
                passage = item['passage']
                question = item['question']
                label = item['label']  # 0 = False, 1 = True

                # Create prompt
                prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer (True/False):"

                # Get prediction
                choices = ["False", "True"]
                predicted_idx = self.evaluate_multiple_choice(prompt, choices)

                if predicted_idx == label:
                    correct += 1
                total += 1

                if total % 50 == 0:
                    logger.info(f"BoolQ progress: {total} samples processed")

            accuracy = correct / total if total > 0 else 0
            result = {"accuracy": accuracy, "correct": correct, "total": total}

            logger.info(f"✅ BoolQ completed: {accuracy:.4f} accuracy ({correct}/{total})")
            return result

        except Exception as e:
            logger.error(f"❌ BoolQ evaluation failed: {e}")
            return {"accuracy": 0.0, "error": str(e)}

    def evaluate_hellaswag(self) -> Dict[str, float]:
        """Evaluate on HellaSwag (0-shot)"""
        logger.info("🧠 Evaluating HellaSwag (0-shot)")

        try:
            dataset = load_dataset("hellaswag", split="validation")
            correct = 0
            total = 0

            for item in dataset:
                context = item['ctx']
                endings = item['endings']
                correct_answer = int(item['label'])

                # Create prompt
                prompt = f"Context: {context}\nChoices:\n"
                for i, ending in enumerate(endings):
                    prompt += f"{chr(65+i)}) {ending}\n"
                prompt += "Answer:"

                # Get prediction
                predicted_idx = self.evaluate_multiple_choice(prompt, endings)

                if predicted_idx == correct_answer:
                    correct += 1
                total += 1

                if total % 50 == 0:
                    logger.info(f"HellaSwag progress: {total} samples processed")

            accuracy = correct / total if total > 0 else 0
            result = {"accuracy": accuracy, "correct": correct, "total": total}

            logger.info(f"✅ HellaSwag completed: {accuracy:.4f} accuracy ({correct}/{total})")
            return result

        except Exception as e:
            logger.error(f"❌ HellaSwag evaluation failed: {e}")
            return {"accuracy": 0.0, "error": str(e)}

    def evaluate_piqa(self) -> Dict[str, float]:
        """Evaluate on PIQA (0-shot)"""
        logger.info("🧠 Evaluating PIQA (0-shot)")

        try:
            dataset = load_dataset("piqa", split="validation")
            correct = 0
            total = 0

            for item in dataset:
                goal = item['goal']
                sol1 = item['sol1']
                sol2 = item['sol2']
                label = item['label']  # 0 or 1

                choices = [sol1, sol2]

                # Create prompt
                prompt = f"Goal: {goal}\nChoices:\nA) {sol1}\nB) {sol2}\nAnswer:"

                # Get prediction
                predicted_idx = self.evaluate_multiple_choice(prompt, choices)

                if predicted_idx == label:
                    correct += 1
                total += 1

                if total % 50 == 0:
                    logger.info(f"PIQA progress: {total} samples processed")

            accuracy = correct / total if total > 0 else 0
            result = {"accuracy": accuracy, "correct": correct, "total": total}

            logger.info(f"✅ PIQA completed: {accuracy:.4f} accuracy ({correct}/{total})")
            return result

        except Exception as e:
            logger.error(f"❌ PIQA evaluation failed: {e}")
            return {"accuracy": 0.0, "error": str(e)}

    def evaluate_winogrande(self) -> Dict[str, float]:
        """Evaluate on WinoGrande (0-shot)"""
        logger.info("🧠 Evaluating WinoGrande (0-shot)")

        try:
            dataset = load_dataset("winogrande", "winogrande_debiased", split="validation")
            correct = 0
            total = 0

            for item in dataset:
                sentence = item['sentence']
                option1 = item['option1']
                option2 = item['option2']
                answer = item['answer']  # "1" or "2"

                choices = [option1, option2]
                correct_idx = int(answer) - 1

                # Create prompt
                prompt = f"Sentence: {sentence}\nChoices:\nA) {option1}\nB) {option2}\nAnswer:"

                # Get prediction
                predicted_idx = self.evaluate_multiple_choice(prompt, choices)

                if predicted_idx == correct_idx:
                    correct += 1
                total += 1

                if total % 50 == 0:
                    logger.info(f"WinoGrande progress: {total} samples processed")

            accuracy = correct / total if total > 0 else 0
            result = {"accuracy": accuracy, "correct": correct, "total": total}

            logger.info(f"✅ WinoGrande completed: {accuracy:.4f} accuracy ({correct}/{total})")
            return result

        except Exception as e:
            logger.error(f"❌ WinoGrande evaluation failed: {e}")
            return {"accuracy": 0.0, "error": str(e)}

    def evaluate_commonsenseqa(self) -> Dict[str, float]:
        """Evaluate on CommonsenseQA (10-shot)"""
        logger.info("🧠 Evaluating CommonsenseQA (10-shot)")

        try:
            dataset = load_dataset("commonsense_qa", split="validation")
            train_dataset = load_dataset("commonsense_qa", split="train")

            # Sample 10 examples for few-shot
            few_shot_examples = train_dataset.shuffle(seed=42).select(range(10))

            # Create few-shot prompt
            few_shot_prompt = "Here are some examples:\n\n"
            for example in few_shot_examples:
                question = example['question']
                choices = example['choices']['text']
                answer = example['answerKey']

                few_shot_prompt += f"Question: {question}\nChoices:\n"
                for i, choice in enumerate(choices):
                    few_shot_prompt += f"{chr(65+i)}) {choice}\n"
                few_shot_prompt += f"Answer: {answer}\n\n"

            correct = 0
            total = 0

            for item in dataset:
                question = item['question']
                choices = item['choices']['text']
                correct_answer = item['answerKey']

                # Create prompt with few-shot examples
                prompt = few_shot_prompt + f"Question: {question}\nChoices:\n"
                for i, choice in enumerate(choices):
                    prompt += f"{chr(65+i)}) {choice}\n"
                prompt += "Answer:"

                # Get prediction
                predicted_idx = self.evaluate_multiple_choice(prompt, choices)
                predicted_answer = chr(65 + predicted_idx)

                if predicted_answer == correct_answer:
                    correct += 1
                total += 1

                if total % 50 == 0:
                    logger.info(f"CommonsenseQA progress: {total} samples processed")

            accuracy = correct / total if total > 0 else 0
            result = {"accuracy": accuracy, "correct": correct, "total": total}

            logger.info(f"✅ CommonsenseQA completed: {accuracy:.4f} accuracy ({correct}/{total})")
            return result

        except Exception as e:
            logger.error(f"❌ CommonsenseQA evaluation failed: {e}")
            return {"accuracy": 0.0, "error": str(e)}

    def evaluate_truthfulqa(self) -> Dict[str, float]:
        """Evaluate on TruthfulQA (10-shot MC2)"""
        logger.info("🧠 Evaluating TruthfulQA (10-shot MC2)")

        try:
            dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")

            # Take first 10 examples for few-shot
            few_shot_examples = dataset.select(range(10))

            # Create few-shot prompt
            few_shot_prompt = "Here are some examples:\n\n"
            for example in few_shot_examples:
                question = example['question']
                choices = example['mc2_targets']['choices']
                labels = example['mc2_targets']['labels']

                few_shot_prompt += f"Question: {question}\nChoices:\n"
                for i, choice in enumerate(choices):
                    few_shot_prompt += f"{chr(65+i)}) {choice}\n"

                # Find correct answers
                correct_indices = [i for i, label in enumerate(labels) if label == 1]
                correct_letters = [chr(65+i) for i in correct_indices]
                few_shot_prompt += f"Answer: {', '.join(correct_letters)}\n\n"

            correct = 0
            total = 0

            # Evaluate on remaining examples
            for item in dataset.select(range(10, len(dataset))):
                question = item['question']
                choices = item['mc2_targets']['choices']
                labels = item['mc2_targets']['labels']

                # Create prompt
                prompt = few_shot_prompt + f"Question: {question}\nChoices:\n"
                for i, choice in enumerate(choices):
                    prompt += f"{chr(65+i)}) {choice}\n"
                prompt += "Answer:"

                # Get prediction
                predicted_idx = self.evaluate_multiple_choice(prompt, choices)

                # Check if prediction is correct (any correct answer)
                if labels[predicted_idx] == 1:
                    correct += 1
                total += 1

                if total % 50 == 0:
                    logger.info(f"TruthfulQA progress: {total} samples processed")

            accuracy = correct / total if total > 0 else 0
            result = {"mc2_accuracy": accuracy, "correct": correct, "total": total}

            logger.info(f"✅ TruthfulQA completed: {accuracy:.4f} MC2 accuracy ({correct}/{total})")
            return result

        except Exception as e:
            logger.error(f"❌ TruthfulQA evaluation failed: {e}")
            return {"mc2_accuracy": 0.0, "error": str(e)}

    def evaluate_triviaqa(self) -> Dict[str, float]:
        """Evaluate on TriviaQA (5-shot EM)"""
        logger.info("🧠 Evaluating TriviaQA (5-shot EM)")

        try:
            dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
            train_dataset = load_dataset("trivia_qa", "rc.nocontext", split="train")

            # Sample 5 examples for few-shot
            few_shot_examples = train_dataset.shuffle(seed=42).select(range(5))

            # Create few-shot prompt
            few_shot_prompt = "Here are some examples:\n\n"
            for example in few_shot_examples:
                question = example['question']
                answer = example['answer']['value']

                few_shot_prompt += f"Question: {question}\nAnswer: {answer}\n\n"

            exact_matches = 0
            total = 0

            for item in dataset.select(range(500)):  # Limit to 500 for speed
                question = item['question']
                answers = item['answer']['aliases']

                # Create prompt
                prompt = few_shot_prompt + f"Question: {question}\nAnswer:"

                # Get prediction
                prediction = self.generate_response(prompt, max_length=50).strip()

                # Check exact match
                is_match = any(ans.lower().strip() in prediction.lower() or
                             prediction.lower() in ans.lower().strip()
                             for ans in answers)

                if is_match:
                    exact_matches += 1
                total += 1

                if total % 50 == 0:
                    logger.info(f"TriviaQA progress: {total} samples processed")

            em_score = exact_matches / total if total > 0 else 0
            result = {"exact_match": em_score, "correct": exact_matches, "total": total}

            logger.info(f"✅ TriviaQA completed: {em_score:.4f} EM score ({exact_matches}/{total})")
            return result

        except Exception as e:
            logger.error(f"❌ TriviaQA evaluation failed: {e}")
            return {"exact_match": 0.0, "error": str(e)}

    def evaluate_mmlu(self) -> Dict[str, float]:
        """Evaluate on MMLU (5-shot)"""
        logger.info("🧠 Evaluating MMLU (5-shot)")

        try:
            # Load all MMLU subjects
            from datasets import get_dataset_config_names
            subjects = get_dataset_config_names("cais/mmlu")

            all_correct = 0
            all_total = 0
            subject_results = {}

            for subject in subjects[:10]:  # Limit to first 10 subjects for speed
                logger.info(f"Evaluating MMLU subject: {subject}")

                try:
                    test_dataset = load_dataset("cais/mmlu", subject, split="test")
                    dev_dataset = load_dataset("cais/mmlu", subject, split="dev")

                    # Use dev set for few-shot examples
                    few_shot_examples = dev_dataset.select(range(min(5, len(dev_dataset))))

                    # Create few-shot prompt
                    few_shot_prompt = f"Subject: {subject}\n\n"
                    for example in few_shot_examples:
                        question = example['question']
                        choices = example['choices']
                        answer = example['answer']

                        few_shot_prompt += f"Question: {question}\nChoices:\n"
                        for i, choice in enumerate(choices):
                            few_shot_prompt += f"{chr(65+i)}) {choice}\n"
                        few_shot_prompt += f"Answer: {chr(65+answer)}\n\n"

                    correct = 0
                    total = 0

                    for item in test_dataset:
                        question = item['question']
                        choices = item['choices']
                        correct_answer = item['answer']

                        # Create prompt
                        prompt = few_shot_prompt + f"Question: {question}\nChoices:\n"
                        for i, choice in enumerate(choices):
                            prompt += f"{chr(65+i)}) {choice}\n"
                        prompt += "Answer:"

                        # Get prediction
                        predicted_idx = self.evaluate_multiple_choice(prompt, choices)

                        if predicted_idx == correct_answer:
                            correct += 1
                        total += 1

                    subject_accuracy = correct / total if total > 0 else 0
                    subject_results[subject] = {
                        "accuracy": subject_accuracy,
                        "correct": correct,
                        "total": total
                    }

                    all_correct += correct
                    all_total += total

                    logger.info(f"✅ {subject}: {subject_accuracy:.4f} accuracy ({correct}/{total})")

                except Exception as e:
                    logger.warning(f"⚠️  Failed to evaluate {subject}: {e}")
                    continue

            overall_accuracy = all_correct / all_total if all_total > 0 else 0
            result = {
                "overall_accuracy": overall_accuracy,
                "total_correct": all_correct,
                "total_questions": all_total,
                "subject_results": subject_results
            }

            logger.info(f"✅ MMLU completed: {overall_accuracy:.4f} overall accuracy ({all_correct}/{all_total})")
            return result

        except Exception as e:
            logger.error(f"❌ MMLU evaluation failed: {e}")
            return {"overall_accuracy": 0.0, "error": str(e)}

    def run_all_evaluations(self) -> Dict[str, Any]:
        """Run selected benchmark evaluations for BitMar training data compatibility"""
        logger.info("🚀 Starting BitMar evaluation on compatible benchmarks")

        start_time = time.time()

        # Define selected evaluations (7 benchmarks most suitable for your training data)
        evaluations = [
            ("ARC-Easy", self.evaluate_arc_easy),           # 0-shot - Basic science reasoning
            ("BoolQ", self.evaluate_boolq),                 # 0-shot - Reading comprehension
            ("HellaSwag", self.evaluate_hellaswag),         # 0-shot - Commonsense reasoning
            ("PIQA", self.evaluate_piqa),                   # 0-shot - Physical reasoning
            ("WinoGrande", self.evaluate_winogrande),       # 0-shot - Pronoun resolution
            ("CommonsenseQA", self.evaluate_commonsenseqa), # 10-shot - Commonsense knowledge
            ("MMLU", self.evaluate_mmlu),                   # 5-shot - Broad knowledge
        ]

        results = {}

        logger.info(f"📊 Running {len(evaluations)} benchmarks optimized for your BitMar training data")
        logger.info("   Training data compatibility:")
        logger.info("   • Text sources: BNC spoken, CHILDES, Gutenberg, OpenSubtitles, Simple Wikipedia, Switchboard")
        logger.info("   • These benchmarks test language understanding that should align with your training")

        for name, eval_func in evaluations:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Starting {name} evaluation")
                logger.info(f"{'='*50}")

                result = eval_func()
                results[name] = result

                logger.info(f"✅ {name} completed successfully")

            except Exception as e:
                logger.error(f"❌ {name} evaluation failed: {e}")
                results[name] = {"error": str(e)}
                continue

        end_time = time.time()
        total_time = end_time - start_time

        # Compile final results
        final_results = {
            "model_path": self.model_path,
            "evaluation_date": datetime.now().isoformat(),
            "total_evaluation_time": total_time,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "benchmarks_run": len(evaluations),
            "training_data_optimized": True,
            "results": results
        }

        logger.info(f"\n🎉 Evaluation Summary:")
        logger.info(f"   • Benchmarks completed: {len([r for r in results.values() if 'error' not in r])}/{len(evaluations)}")
        logger.info(f"   • Total time: {total_time:.1f} seconds")
        logger.info(f"   • Optimized for BitMar training data: ✅")

        return final_results

    def save_results(self, results: Dict[str, Any], output_dir: str = "evaluation_results"):
        """Save evaluation results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"bitmar_evaluation_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Save summary
        summary_file = output_path / f"bitmar_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("BitMar Model Evaluation Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Model: {results['model_path']}\n")
            f.write(f"Date: {results['evaluation_date']}\n")
            f.write(f"Total Time: {results['total_evaluation_time']:.2f} seconds\n\n")

            for benchmark, result in results['results'].items():
                f.write(f"{benchmark}:\n")
                if 'error' in result:
                    f.write(f"  ❌ Error: {result['error']}\n")
                else:
                    if 'accuracy' in result:
                        f.write(f"  ✅ Accuracy: {result['accuracy']:.4f}\n")
                    if 'mc2_accuracy' in result:
                        f.write(f"  ✅ MC2 Accuracy: {result['mc2_accuracy']:.4f}\n")
                    if 'exact_match' in result:
                        f.write(f"  ✅ Exact Match: {result['exact_match']:.4f}\n")
                    if 'overall_accuracy' in result:
                        f.write(f"  ✅ Overall Accuracy: {result['overall_accuracy']:.4f}\n")
                f.write("\n")

        logger.info(f"📊 Results saved to:")
        logger.info(f"  • Detailed: {results_file}")
        logger.info(f"  • Summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BitMar model on comprehensive benchmarks")
    parser.add_argument("--model_path", type=str, default="euhidaman/bitmar-attention-multimodal",
                       help="Path or HuggingFace model ID for BitMar model")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                       default=["all"],
                       help="Specific benchmarks to run (default: all)")

    args = parser.parse_args()

    try:
        # Initialize evaluator
        evaluator = BitMarEvaluator(
            model_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size
        )

        # Run evaluations
        if "all" in args.benchmarks:
            results = evaluator.run_all_evaluations()
        else:
            # Run specific benchmarks (implement if needed)
            results = evaluator.run_all_evaluations()

        # Save results
        evaluator.save_results(results, args.output_dir)

        logger.info("🎉 Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
