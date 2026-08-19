# Reproducing CLIP for Zero-Shot Image Classification with Fairness Analysis

## 📌 Project Overview
Modern vision models typically require large labeled datasets. The CLIP model (Contrastive Language-Image Pretraining) addresses this by learning from (image, text) pairs, enabling zero-shot image classification. 

This project reproduces OpenAI's core CLIP baseline results and extends the original research by evaluating the model's fairness across demographic subgroups (Gender and Race) using the full FairFace validation dataset (N = 10,954). 

## 📂 Repository Structure
* `/notebooks/CLIP_Fairness_Evaluation.ipynb` - Interactive Jupyter Notebook containing all evaluations, full-dataset confusion matrices, and the bias mitigation experiments.
* `/src/cifar10_baseline.py` - Python script evaluating baseline zero-shot accuracy on CIFAR-10.
* `/src/fairface-setup.py` - Script to download and inspect the FairFace dataset metadata.
* `/src/fairface-eval.py` - Script to evaluate zero-shot demographic fairness on FairFace.
* `/src/test_clip.py` - Script for local testing and environment verification.
* `/src/zeroshot-test.py` - Smoke test to verify CLIP model loading and setup.
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

- **CIFAR-10 Baseline:** Achieved **88.80%** top-1 zero-shot accuracy on the CIFAR-10 test set (N = 10,000), establishing a functional baseline for the ViT-B/32 CLIP inference pipeline.

- **Gender Evaluation:** Achieved **94.60%** overall accuracy on binary gender classification across 10,954 FairFace validation images. Male accuracy was **94.55%** and female accuracy was **94.65%**, corresponding to a **0.10 percentage-point disparity gap**.

- **Race Evaluation:** Achieved **64.86%** overall accuracy across seven racial categories. Subgroup performance ranged from **49.30% (White)** to **84.90% (Black)**, resulting in a **35.60 percentage-point disparity gap**.

- **Intersectional Prompt Mitigation:** Adding gender context to race prompts (e.g., *"a face photo of a Black woman"*) increased overall race-classification accuracy from **64.86% to 66.72%**, an improvement of **1.86 percentage points**.

- **Key Finding:** The results suggest that richer textual context can improve zero-shot demographic classification, but substantial subgroup performance disparities remain after the tested prompt-conditioning intervention.
