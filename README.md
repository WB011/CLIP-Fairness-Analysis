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
* **Baseline Accuracy:** Achieved **88.80%** top-1 zero-shot accuracy on CIFAR-10 (N = 10,000), establishing a functional baseline for the ViT-B/32 architecture[cite: 1].
* **Gender Fairness:** Achieved **94.60%** overall accuracy on binary gender classification across 10,954 images. The model demonstrated excellent demographic parity, with a negligible disparity gap of **0.10** percentage points (94.55% for Male vs. 94.65% for Female).
* **Racial Bias & Mitigation:** The baseline zero-shot accuracy for race classification across 7 categories was evaluated at **64.86%**[cite: 2]. While this establishes a baseline capacity for demographic categorization, it leaves a significant margin for misclassification. Applying an intersectional prompt mitigation strategy (e.g., providing explicit gender context like "a face photo of a Black woman") successfully improved the overall accuracy to **66.72%**[cite: 2]. 
* **The Architecture Ceiling:** Despite the mitigation, a severe fairness disparity gap of **35.60** percentage points remained between the highest-performing group (Black, 84.90%) and the lowest (White, 49.30%). This demonstrates that while textual context helps reduce representational ambiguity, lightweight zero-shot models like ViT-B/32 possess deep representational limits and should not be deployed for sensitive tasks without targeted fine-tuning.
