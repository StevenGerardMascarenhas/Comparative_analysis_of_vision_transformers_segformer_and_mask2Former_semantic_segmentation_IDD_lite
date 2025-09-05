

---

# Comparative Analysis of Vision Transformer Architectures for Semantic Segmentation

**SegFormer vs Mask2Former on the Indian Driving Dataset (IDD-Lite)**

📍 MSc AIML Dissertation – University of Limerick (2024–2025)

---

## 📖 Overview

This repository contains the implementation and experiments from my MSc dissertation:

> **Comparative Analysis of Vision Transformer Architectures: SegFormer and Mask2Former for Semantic Segmentation in Unstructured Road Environments using IDD-Lite.**

The project evaluates two state-of-the-art Vision Transformer models — **SegFormer** and **Mask2Former** — on the **IDD-Lite dataset**, which represents the challenging unstructured road conditions typical of Indian traffic.

The implementation was built using:

* [🤗 Hugging Face Transformers](https://huggingface.co/transformers/)
* [PyTorch](https://pytorch.org/)
* [Albumentations](https://albumentations.ai/)

------
 My Full MSc dissertation write-up:

**MscAIML_Dissertation_24044407_Steven_Gerard_Mascarenhas.pdf**
https://github.com/StevenGerardMascarenhas/Comparative_analysis_of_vision_transformers_segformer_and_mask2Former_semantic_segmentation_IDD_lite/blob/main/MscAIML_Dissertation_24044407_Steven_Gerard_Mascarenhas.pdf

It contains the detailed research study, methodology, experiments, and results that this codebase is built on.

---
## 📂 Dataset (IDD-Lite)

The experiments use **IDD-Lite**, a subset of the [Indian Driving Dataset (IDD)](https://idd.insaan.iiit.ac.in/).

### 🔽 Download Instructions

1. Download IDD-Lite from the [official website](https://idd.insaan.iiit.ac.in/).



## ⚙️ Setup

### 1. Clone Repository

```bash
git clone https://github.com/StevenGerardMascarenhas/Comparative_analysis_of_vision_transformers_segformer_and_mask2Former_semantic_segmentation_IDD_lite.git
cd Comparative_analysis_of_vision_transformers_segformer_and_mask2Former_semantic_segmentation_IDD_lite
```

### 2. Create Environment

```bash
conda create -n iddseg python=3.10
conda activate iddseg
pip install -r requirements.txt
```

### 3. Download Dataset

Follow [dataset instructions](#-dataset-idd-lite).

---

## 🚀 Running Experiments

Each notebook corresponds to an experiment.

* **SegFormer**

  * `segformer_b0_finetuned_cityscapes_with_data_augmentation.ipynb`
  * `segformer_b1_train_from_scratch_with_data_augmentation.ipynb`
  * `segformer_b2_finetuned_cityscapes_with_data_augmentation_focal_loss.ipynb`

* **Mask2Former**

  * `mask2former_swin_tiny_finetuned_cityscapes_with_data_augmentation.ipynb`
  * `mask2former_swin_small_trained_from_scratch_with_data_augmentation.ipynb`

Example (run with Jupyter):

```bash
jupyter notebook segformer_b2_finetuned_cityscapes_with_data_augmentation_focal_loss.ipynb
```

---

## 📊 some of the Results (Highlights)

| Model                         | Training   | mIoU      | Pixel Accuracy | FPS  | Notes                  |
| ----------------------------- | ---------- | --------- | -------------- | ---- | ---------------------- |
| **SegFormer B0**              | Pretrained | 0.701     | 0.902          | 14.5 | Fast baseline          |
| **SegFormer B2**              | Pretrained | 0.723     | 0.913          | 10.7 | Best SegFormer         |
| **SegFormer B2** + Focal Loss | Pretrained | 0.742     | 0.921          | 10.4 | Best SegFormer variant |
| **Mask2Former Tiny**          | Pretrained | 0.748     | 0.926          | 8.1  | Strong overall         |
| **Mask2Former Small**         | Pretrained |  0.761   | 0.932           | 7.4  | Best accuracy overall  |
| Scratch (any model)           | Scratch    | <0.35     | <0.70          | –    | Failed to generalize   |

### Key Findings

* **Transfer learning is essential**: Models trained from scratch failed (IoU=0.0 on some classes).
* **Mask2Former Swin-Small** delivered the **highest accuracy**.
* **SegFormer B2 + Focal Loss** offered the **best efficiency/accuracy balance**.

---

## 🤗 Hugging Face Models Used

* `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`
* `nvidia/segformer-b1-finetuned-cityscapes-1024-1024`
* `nvidia/segformer-b2-finetuned-cityscapes-1024-1024`
* `facebook/mask2former-swin-tiny-cityscapes-semantic`
* `facebook/mask2former-swin-small-cityscapes-semantic`

All models were loaded via Hugging Face `from_pretrained(...)`.

---

## 🤗 Hugging Face Documentation

* SegFormer: [https://huggingface.co/docs/transformers/en/model\_doc/segformer](https://huggingface.co/docs/transformers/en/model_doc/segformer)
* Mask2Former: [https://huggingface.co/docs/transformers/en/model\_doc/mask2former](https://huggingface.co/docs/transformers/en/model_doc/mask2former)

---

**Steven Gerard Mascarenhas**,
*MSc Dissertation: Comparative Analysis of Vision Transformer Architectures for Semantic Segmentation on IDD-Lite*,
University of Limerick, 2025.

---

## 🙌 Acknowledgements

* Supervisor: **Dr. Emil Vassev**, University of Limerick
* [🤗 Hugging Face](https://huggingface.co/)
* [Albumentations](https://albumentations.ai/)
* [PyTorch](https://pytorch.org/)

---




