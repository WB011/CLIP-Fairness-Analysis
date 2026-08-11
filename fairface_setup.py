"""
FairFace Dataset Initialization and Inspection

This script downloads the FairFace dataset from Hugging Face
and inspects the available demographic labels to prepare for the fairness evaluation.
"""

import logging
import sys
from datasets import load_dataset

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def main() -> None:
    """Main execution function to load and inspect FairFace."""
    logger.info("Downloading and loading FairFace dataset from Hugging Face...")
    
    # Load the train split of the FairFace dataset
    try:
        dataset = load_dataset("HuggingFaceM4/FairFace", "1.25", split="train")
        logger.info(f"Successfully loaded {len(dataset)} images from the training split.")
        
        # Inspect the first record's metadata (skipping the heavy image object)
        sample_record = dataset[0]
        
        logger.info("--- Sample Record Metadata ---")
        for key, value in sample_record.items():
            if key != "image":
                logger.info(f"{key.capitalize():<10} : {value}")
                
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")

if __name__ == "__main__":
    main()