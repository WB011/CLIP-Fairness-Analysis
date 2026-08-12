# Reproducing CLIP for Zero-Shot Image Classification with Fairness Analysis

## 📌 Project Overview
Modern vision models typically require large labeled datasets. The CLIP model (Contrastive Language-Image Pretraining) addresses this by learning from (image, text) pairs, enabling zero-shot image classification. 

This project reproduces OpenAI's core CLIP baseline results and extends the original research by evaluating the model's fairness across demographic subgroups (Gender and Race) using the full FairFace dataset ($N = 10,954$). 

## 📂 Repository Structure
* `/notebooks/CLIP_Fairness_Evaluation.ipynb` - Interactive Jupyter Notebook containing all visualizations, full-dataset confusion matrices, and the bias mitigation experiment.
* `/src/zero_shot_test.py` - Sanity check script for model initialization.
* `/src/cifar10_baseline.py` - Evaluates baseline zero-shot accuracy on CIFAR-10.
* `/src/fairface_setup.py` - Downloads and initializes the FairFace dataset.
* `/src/fairface_eval.py` - Core script for demographic fairness evaluation.
* `/results/` - Contains JSON metric outputs and confusion matrix visualizations.

## 🚀 Installation & Setup
1. Clone this repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   * Windows: `venv\Scripts\activate`
   * Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## 📊 Key Findings
* **Baseline Accuracy:** Achieved **88.80%** top-1 zero-shot accuracy on CIFAR-10, establishing a functional baseline for the ViT-B/32 architecture.
* **Gender Fairness:** Achieved **94.60%** accuracy on binary gender classification, demonstrating high utility and minimal disparity.
* **Racial Bias & Mitigation:** Discovered severe systematic biases and complete task failure in zero-shot racial classification (**2.85%** accuracy). Attempting bias mitigation via intersectional prompt engineering (e.g., "a face photo of a Black man") yielded no improvement (**2.84%**), indicating deep pre-training biases within the ViT-B/32 weights rather than surface-level prompt formatting issues.
