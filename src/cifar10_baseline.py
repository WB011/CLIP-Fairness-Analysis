"""
CLIP Zero-Shot Baseline Evaluation on CIFAR-10

This script evaluates the zero-shot classification accuracy of pre-trained CLIP models
on the CIFAR-10 test dataset, establishing the reproduction baseline for the project.
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Tuple

import torch
import clip
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="CLIP Zero-Shot Baseline Evaluation on CIFAR-10"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B/32",
        help="CLIP backbone architecture (e.g., ViT-B/32, RN50)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="DataLoader batch size",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of test images to evaluate (useful for quick CPU runs).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory to download/store CIFAR-10 dataset",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="cifar10_baseline_results.json",
        help="File path to save result metrics",
    )
    return parser.parse_args()


def get_device() -> torch.device:
    """Detect and return available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Compute device: CUDA GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logger.info("Compute device: CPU")
    return device


def load_clip_and_dataset(
    model_name: str, data_dir: str, device: torch.device
) -> Tuple[torch.nn.Module, callable, torchvision.datasets.CIFAR10]:
    """
    Load CLIP model and initialize CIFAR-10 test set with CLIP preprocessing.
    """
    logger.info(f"Loading CLIP backbone model: '{model_name}'...")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    logger.info("Downloading/Loading CIFAR-10 test set...")
    cifar10_test = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=preprocess
    )
    
    return model, preprocess, cifar10_test


def compute_text_embeddings(
    model: torch.nn.Module, class_names: List[str], device: torch.device
) -> torch.Tensor:
    """
    Construct prompt templates and pre-compute normalized text feature vectors.
    """
    logger.info("Constructing prompt embeddings for CIFAR-10 classes...")
    # Standard prompt template used in OpenAI CLIP paper
    prompts = [f"a photo of a {class_name}" for class_name in class_names]
    text_tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        # Normalize vectors for cosine similarity
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features


def evaluate_zero_shot(
    model: torch.nn.Module,
    dataloader: DataLoader,
    text_features: torch.Tensor,
    device: torch.device,
    max_samples: int = None,
) -> Tuple[float, int]:
    """
    Run evaluation loop over test dataset and compute Top-1 Accuracy.
    """
    correct_predictions = 0
    total_processed = 0

    logger.info("Starting zero-shot evaluation loop...")
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images = images.to(device)
            labels = labels.to(device)

            # Compute image embeddings
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Cosine similarity matrix multiplication
            similarity_logits = (100.0 * image_features @ text_features.T)
            predictions = similarity_logits.argmax(dim=-1)

            # Accumulate accurate counts
            correct_predictions += (predictions == labels).sum().item()
            total_processed += labels.size(0)

            # Stop early if max_samples budget reached
            if max_samples and total_processed >= max_samples:
                logger.info(f"Reached max sample limit: {max_samples} images.")
                break

    accuracy = (correct_predictions / total_processed) * 100.0
    return accuracy, total_processed


def main() -> None:
    args = parse_args()
    device = get_device()

    # 1. Prepare Model and Dataset
    model, preprocess, cifar10_test = load_clip_and_dataset(
        args.model_name, args.data_dir, device
    )

    dataloader = DataLoader(
        cifar10_test, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # 2. Embed Prompt Vector space
    class_names = cifar10_test.classes
    text_features = compute_text_embeddings(model, class_names, device)

    # 3. Execute Benchmark
    top1_accuracy, total_samples = evaluate_zero_shot(
        model, dataloader, text_features, device, max_samples=args.max_samples
    )

    # 4. Print and Save Summary
    logger.info("--- Evaluation Complete ---")
    logger.info(f"Model Architecture : {args.model_name}")
    logger.info(f"Evaluated Samples  : {total_samples}")
    logger.info(f"Top-1 Accuracy     : {top1_accuracy:.2f}%")

    results = {
        "model_name": args.model_name,
        "dataset": "CIFAR-10",
        "eval_samples": total_samples,
        "top1_accuracy": top1_accuracy,
    }

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to '{args.output_json}'.")


if __name__ == "__main__":
    main()
