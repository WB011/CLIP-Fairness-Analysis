import argparse
import logging
import sys
from typing import Dict, List, Tuple

import torch
import clip
from PIL import Image

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CLIP Zero-Shot Verification & Smoke Test"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B/32",
        help="CLIP backbone architecture (e.g., ViT-B/32, RN50)",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to a local image file. Generates synthetic image if omitted.",
    )
    return parser.parse_args()


def get_device() -> torch.device:
    """Detect and return the best available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Compute device: CUDA GPU ({gpu_name})")
    else:
        device = torch.device("cpu")
        logger.info("Compute device: CPU")
    return device


def load_clip_model(model_name: str, device: torch.device) -> Tuple[torch.nn.Module, callable]:
    """
    Load pre-trained CLIP model weights and preprocessing transform pipeline.
    
    Args:
        model_name (str): The name of the CLIP variant to load.
        device (torch.device): Compute device target.

    Returns:
        Tuple[torch.nn.Module, callable]: Loaded PyTorch model and preprocess function.
    """
    logger.info(f"Loading CLIP model checkpoint '{model_name}'...")
    try:
        model, preprocess = clip.load(model_name, device=device)
        model.eval()  # Set model to evaluation mode
        logger.info("CLIP model successfully loaded into memory.")
        return model, preprocess
    except Exception as e:
        logger.error(f"Failed to load CLIP model '{model_name}': {e}")
        sys.exit(1)


def load_or_generate_image(image_path: str = None) -> Image.Image:
    """
    Load an image from disk or construct a synthetic solid test image.

    Args:
        image_path (str, optional): Filepath to image. Defaults to None.

    Returns:
        Image.Image: PIL RGB Image object.
    """
    if image_path:
        logger.info(f"Loading user image from: {image_path}")
        return Image.open(image_path).convert("RGB")

    logger.info("No image path provided. Generating synthetic red test sample...")
    return Image.new("RGB", (224, 224), color=(255, 0, 0))


def zero_shot_classify(
    model: torch.nn.Module,
    preprocess: callable,
    image: Image.Image,
    candidate_labels: List[str],
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute zero-shot class probabilities for a given image across candidate textual labels.

    Args:
        model (torch.nn.Module): CLIP model.
        preprocess (callable): Vision preprocessing pipeline.
        image (Image.Image): Input image.
        candidate_labels (List[str]): List of raw concept names (e.g., ["dog", "cat"]).
        device (torch.device): Compute device target.

    Returns:
        Dict[str, float]: Mapping of candidate label to prediction probability.
    """
    # 1. Preprocess and create batch dimension
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # 2. Format textual prompts and tokenize
    prompts = [f"a photo of a {label}" for label in candidate_labels]
    text_tokens = clip.tokenize(prompts).to(device)

    # 3. Compute embeddings without tracking gradients
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

        # L2-normalize vectors for cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Calculate cosine similarity scaled by logit scale factor
        similarity_logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probabilities = similarity_logits.squeeze(0).cpu().numpy()

    return {label: float(prob) for label, prob in zip(candidate_labels, probabilities)}


def main() -> None:
    """Main execution function."""
    args = parse_args()
    device = get_device()
    model, preprocess = load_clip_model(args.model_name, device)

    # Define test payload
    image = load_or_generate_image(args.image_path)
    candidate_labels = ["red square", "dog", "automobile", "blue circle"]

    # Execute inference
    logger.info("Executing zero-shot forward pass...")
    results = zero_shot_classify(model, preprocess, image, candidate_labels, device)

    # Output formatted results
    logger.info("--- Zero-Shot Inference Results ---")
    for label, prob in results.items():
        logger.info(f"Class: 'a photo of a {label:<12}' | Confidence: {prob * 100:6.2f}%")


if __name__ == "__main__":
    main()