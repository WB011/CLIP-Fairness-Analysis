# Reproducing CLIP for Zero-Shot Image Classification with Fairness Analysis

## 📌 Project Overview
Modern vision models typically require large labeled datasets. The CLIP model (Contrastive Language-Image Pretraining) addresses this by learning from (image, text) pairs, enabling zero-shot image classification. 

This project reproduces OpenAI's core CLIP baseline results and extends the original research by evaluating the model's fairness across demographic subgroups (Gender and Race) using the full FairFace validation dataset (N = 10,954). 

## 📂 Repository Structure
* `/notebooks/CLIP_Fairness_Evaluation.ipynb` - Interactive Jupyter Notebook containing all evaluations, full-dataset confusion matrices, and the bias mitigation experiment.
* `/src/cifar10_baseline.py` - Python script evaluating baseline zero-shot accuracy on CIFAR-10.
* `/src/fairface-setup.py` - Script to download and inspect the FairFace dataset metadata.
* `/src/fairface-eval.py` - Script to evaluate zero-shot demographic fairness on FairFace.
* `/src/zeroshot-test.py` - Smoke test to verify CLIP model loading and environment setup.
* `/results/cifar10_baseline_results.json` - JSON output containing the CIFAR-10 baseline metrics.
* `/results/fairface_gender_results.json` - JSON output for Gender fairness metrics.
* `/results/fairface_race_results.json` - JSON output for Race fairness metrics.
* `requirements.txt` - Required Python dependencies for reproducibility.

## 🚀 Installation & Setup
1. Clone this repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   * Windows: `venv\Scripts\activate`
   * Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## 📊 Key Findings
* **Baseline Accuracy:** Achieved **88.80%** top-1 zero-shot accuracy on CIFAR-10 (N = 10,000), establishing a functional baseline for the ViT-B/32 architecture.
* **Gender Fairness:** Achieved **94.60%** accuracy on binary gender classification across 10,954 images, demonstrating high utility and minimal disparity (0.10 pp gap).
* **Racial Bias & Mitigation:** Discovered severe systematic biases and task degradation in zero-shot racial classification (**2.85%** accuracy). Attempting bias mitigation via intersectional prompt engineering (e.g., "a face photo of a Black man") yielded no meaningful improvement (**2.84%**), indicating deep pre-training biases within the representations rather than surface-level prompt formatting issues.
