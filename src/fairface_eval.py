"""
FairFace Dataset Evaluation and Bias Mitigation

This script evaluates the zero-shot classification capabilities of CLIP 
on the FairFace validation dataset for both Gender and Race, and tests 
intersectional prompting for bias mitigation.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Tuple, List

import torch
import clip
from datasets import load_dataset
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
        description="CLIP Fairness Evaluation on FairFace"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B/32",
        help="CLIP backbone architecture",
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
        help="Maximum number of test images to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save the results JSON files",
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


def evaluate_demographic(
    model: torch.nn.Module,
    dataloader: DataLoader,
    target_attr: str,
    labels_list: List[str],
    prompts: List[str],
    device: torch.device,
    max_samples: int = None
) -> Tuple[float, Dict[str, float], int]:
    """Evaluates CLIP on a specific demographic attribute."""
    logger.info(f"Evaluating {target_attr.upper()}...")

    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    group_correct = {label: 0 for label in labels_list}
    group_total = {label: 0 for label in labels_list}
    total_correct, total_samples = 0, 0

    with torch.no_grad():
        for images, races, genders in tqdm(dataloader, desc=f"Evaluating {target_attr}", unit="batch"):
            images = images.to(device)
            labels = races.to(device) if target_attr == "race" else genders.to(device)

            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T)
            predictions = similarity.argmax(dim=-1)
            matches = predictions == labels

            for match, label_idx in zip(matches, labels.cpu().numpy()):
                group_name = labels_list[label_idx]
                group_total[group_name] += 1
                if match.item():
                    group_correct[group_name] += 1
                    total_correct += 1
                total_samples += 1

            if max_samples and total_samples >= max_samples:
                break

    overall_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    group_accs = {
        g: (group_correct[g] / group_total[g] * 100) if group_total[g] > 0 else 0 
        for g in labels_list
    }
    
    return overall_acc, group_accs, total_samples


def evaluate_mitigation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    race_labels: List[str],
    device: torch.device,
    max_samples: int = None
) -> float:
    """Evaluates intersectional bias mitigation (Gender + Race)."""
    logger.info("Testing Bias Mitigation: Adding explicit gender context to prompts...")

    male_prompts = [f"a face photo of a {r.lower()} man" for r in race_labels]
    female_prompts = [f"a face photo of a {r.lower()} woman" for r in race_labels]

    male_tokens = clip.tokenize(male_prompts).to(device)
    female_tokens = clip.tokenize(female_prompts).to(device)

    with torch.no_grad():
        male_features = model.encode_text(male_tokens)
        male_features /= male_features.norm(dim=-1, keepdim=True)

        female_features = model.encode_text(female_tokens)
        female_features /= female_features.norm(dim=-1, keepdim=True)

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, races, genders in tqdm(dataloader, desc="Testing Mitigation", unit="batch"):
            images = images.to(device)
            races = races.to(device)

            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            for i in range(images.size(0)):
                true_race = races[i].item()
                true_gender = genders[i].item()  # 0 is Male, 1 is Female

                text_features = male_features if true_gender == 0 else female_features
                img_feat = image_features[i].unsqueeze(0)

                similarity = (100.0 * img_feat @ text_features.T)
                prediction = similarity.argmax(dim=-1).item()

                if prediction == true_race:
                    total_correct += 1
                total_samples += 1

                if max_samples and total_samples >= max_samples:
                    break
            if max_samples and total_samples >= max_samples:
                break

    return (total_correct / total_samples) * 100 if total_samples > 0 else 0


def main() -> None:
    args = parse_args()
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading CLIP model: '{args.model_name}'...")
    model, preprocess = clip.load(args.model_name, device=device)
    model.eval()

    def fairface_collate(batch):
        images = torch.stack([preprocess(item["image"].convert("RGB")) for item in batch])
        races = torch.tensor([item["race"] for item in batch])
        genders = torch.tensor([item["gender"] for item in batch])
        return images, races, genders

    logger.info("Loading FairFace validation dataset...")
    fairface_val = load_dataset("HuggingFaceM4/FairFace", "1.25", split="validation")
    fairface_loader = DataLoader(
        fairface_val, batch_size=args.batch_size, shuffle=False, collate_fn=fairface_collate
    )

    # 1. Evaluate Gender
    gender_labels = ["Male", "Female"]
    gender_prompts = [f"a face photo of a {g.lower()} person" for g in gender_labels]
    gender_acc, gender_group_accs, total_samples = evaluate_demographic(
        model, fairface_loader, "gender", gender_labels, gender_prompts, device, args.max_samples
    )

    # 2. Evaluate Race Baseline
    race_labels = [
        "East Asian", "Indian", "Black", "White", 
        "Middle Eastern", "Latino or Hispanic", "Southeast Asian"
    ]
    race_prompts = [f"a face photo of a {r} person" for r in race_labels]
    race_acc, race_group_accs, _ = evaluate_demographic(
        model, fairface_loader, "race", race_labels, race_prompts, device, args.max_samples
    )

    # 3. Evaluate Race Mitigation
    mitigated_acc = evaluate_mitigation(
        model, fairface_loader, race_labels, device, args.max_samples
    )

    # 4. Save JSON Results
    gender_results = {
        "model": args.model_name,
        "target_attribute": "gender",
        "overall_accuracy": round(gender_acc, 2),
        "subgroup_accuracies": {k: round(v, 2) for k, v in gender_group_accs.items()},
        "eval_samples": total_samples,
        "fairness_disparity_gap": round(abs(gender_group_accs["Male"] - gender_group_accs["Female"]), 2)
    }

    race_results = {
        "model": args.model_name,
        "target_attribute": "race",
        "overall_accuracy": round(race_acc, 2),
        "mitigated_accuracy": round(mitigated_acc, 2),
        "eval_samples": total_samples,
        "subgroup_accuracies": {k: round(v, 2) for k, v in race_group_accs.items()},
        "fairness_disparity_gap": round(max(race_group_accs.values()) - min(race_group_accs.values()), 2)
    }

    with open(os.path.join(args.output_dir, "fairface_gender_results.json"), "w") as f:
        json.dump(gender_results, f, indent=4)
    
    with open(os.path.join(args.output_dir, "fairface_race_results.json"), "w") as f:
        json.dump(race_results, f, indent=4)

    logger.info("Evaluation complete. Results saved successfully.")


if __name__ == "__main__":
    main()
