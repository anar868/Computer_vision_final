# CXR-CLIP Pneumonia Classification and XAI Workflow

This repository is based on the official PyTorch implementation of CXR-CLIP: Toward Large Scale Chest X-ray Language-Image Pre-training.

Original repository: Soombit-ai/cxr-clip  
Original paper: "CXR-CLIP: Toward Large Scale Chest X-ray Language-Image Pre-training" [arXiv]

This fork extends the original repository with additional workflows for:

- RSNA Pneumonia classification
- NIH ChestX-ray14 Pneumonia vs Normal classification
- Fine-tuned classifier evaluation
- Zero-shot evaluation
- Grad-CAM explainability
- Integrated Gradients explainability
- Reusable extracted dataset folders to avoid repeated Kaggle downloads

---

## 1. Environment setup

The original implementation was tested with:

- PyTorch 1.12
- CUDA 11

Install requirements:

bash pip install -r requirements.txt 

For Colab, Python 3.10 was used in this project.

---

## 2. Repository additions in this fork

This fork adds or modifies the following main files:

text configs/data_train/rsna_pneumonia.yaml configs/data_valid/rsna_pneumonia.yaml configs/data_test/rsna_pneumonia.yaml  configs/data_train/nih_pneumonia.yaml configs/data_valid/nih_pneumonia.yaml configs/data_test/nih_pneumonia.yaml  evaluate_clip_rsna.py evaluate_clip_nih_pneumonia.py evaluate_finetune_nih_pneumonia.py  save_xai_gradcam_results_rsna.py save_xai_ig_results_rsna.py save_xai_gradcam_results_nih.py save_xai_ig_results_nih.py 

The following original files were also modified:

text cxrclip/model/image_classification.py cxrclip/evaluator.py cxrclip/prompt/constants.py requirements.txt 

Main changes include:

- Support for nih_pneumonia
- Safer ResNet checkpoint loading when pretrained checkpoints contain resnet.fc.*
- Evaluation support for the NIH pneumonia binary classification task
- Dataset configs pointing to extracted dataset folders
- XAI scripts for saving Grad-CAM and Integrated Gradients maps

---

## 3. Dataset preparation

Large datasets are not included in this repository.

This project uses extracted local dataset folders:

text /content/datasets/RSNA_Extracted /content/datasets/NIH_Extracted 

These folders are created once from the original Kaggle/NIH data and then saved to Google Drive to avoid repeated large downloads.

### 3.1 RSNA extracted dataset

Expected structure:

text /content/datasets/RSNA_Extracted/   images/   csv/     rsna_train.csv     rsna_valid.csv     rsna_test.csv     rsna_train_unbalanced.csv     rsna_valid_unbalanced.csv     rsna_test_unbalanced.csv     stage_2_train_labels.csv     stage_2_detailed_class_info.csv   manifest.json 

The balanced RSNA split used in this project contains:

text Train: 8416 images   Pneumonia: 4208   Normal:    4208  Test: 1804 images   Pneumonia: 902   Normal:    902 

The extracted RSNA split was verified to match the previously used RSNA training/test IDs.

### 3.2 NIH extracted dataset

Expected structure:

text /content/datasets/NIH_Extracted/   images/   csv/     nih_pneumonia_train.csv     nih_pneumonia_valid.csv     nih_pneumonia_test.csv     Data_Entry_2017.csv     BBox_List_2017.csv   manifest.json 

The NIH dataset is constructed as a binary task:

text Pneumonia vs Normal / No Finding 

The NIH test set is shared for both classification evaluation and XAI:

text Test: 240 images   Pneumonia: 120 images with bounding boxes   Normal:    120 images without bounding boxes 

For XAI, Pneumonia images have ground-truth boxes. Normal images do not have boxes, which is expected.

---

## 4. Model checkpoints

Model checkpoints are not included in this repository.

The original CXR-CLIP pretrained checkpoints can be downloaded from the official links:

| Model / Dataset | M | M,C | M,C,C14 |
|---|---|---|---|
| ResNet50 | Link | Link | Link |
| SwinTiny | Link | Link | Link |

In this project, the following naming convention was used locally:

text models/mcc/r50_m_pretrained.tar   Original pretrained CXR-CLIP checkpoint used for zero-shot evaluation and fine-tuning.  models/mcc/r50_m.tar   RSNA fine-tuned classifier checkpoint.  outputs/.../checkpoints/model-best.tar   Newly fine-tuned classifier checkpoints, for example NIH fine-tuned models. 

---

## 5. Zero-shot evaluation

### 5.1 RSNA zero-shot evaluation

Use the pretrained CXR-CLIP checkpoint:

bash python evaluate_clip_rsna.py \   test.checkpoint=models/mcc/r50_m_pretrained.tar 

### 5.2 NIH zero-shot evaluation

Use the pretrained CXR-CLIP checkpoint:

bash python evaluate_clip_nih_pneumonia.py \   test.checkpoint=models/mcc/r50_m_pretrained.tar 

---

## 6. Fine-tuned classifier evaluation

### 6.1 Evaluate RSNA fine-tuned model on RSNA

bash python evaluate_finetune.py \   test.checkpoint=models/mcc/r50_m.tar 

### 6.2 Evaluate RSNA fine-tuned model on NIH

bash python evaluate_finetune_nih_pneumonia.py \   test.checkpoint=models/mcc/r50_m.tar 

### 6.3 Evaluate NIH fine-tuned model on NIH

Replace the checkpoint path with the actual NIH fine-tuned checkpoint:

bash python evaluate_finetune_nih_pneumonia.py \   test.checkpoint=outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/model-best.tar 

---

## 7. Fine-tuning

### 7.1 Fine-tune on RSNA

Use a pretrained CXR-CLIP checkpoint as the backbone:

bash python finetune.py \   data_train=rsna_pneumonia \   data_valid=rsna_pneumonia \   model.load_backbone_weights=models/mcc/r50_m_pretrained.tar \   model.classifier.config.n_class=1 \   scheduler.config.total_epochs=10 \   scheduler.config.warmup_epochs=0 \   dataloader.train.batch_size=64 \   dataloader.valid.batch_size=64 \   dataloader.train.num_workers=2 \   dataloader.valid.num_workers=2 \   dataloader.train.prefetch_factor=2 \   dataloader.valid.prefetch_factor=2 

### 7.2 Fine-tune on NIH

bash python finetune.py \   data_train=nih_pneumonia \   data_valid=nih_pneumonia \   model.load_backbone_weights=models/mcc/r50_m_pretrained.tar \   model.classifier.config.n_class=1 \   scheduler.config.total_epochs=10 \   scheduler.config.warmup_epochs=0 \   dataloader.train.batch_size=64 \   dataloader.valid.batch_size=64 \   dataloader.train.num_workers=2 \   dataloader.valid.num_workers=2 \   dataloader.train.prefetch_factor=2 \   dataloader.valid.prefetch_factor=2 

---

## 8. Explainability / XAI

Two XAI methods are included:

1. Grad-CAM
2. Integrated Gradients

The scripts save heatmaps as .npy files and save a metadata.csv file with:

- image path
- ground-truth label
- predicted label
- predicted probability
- correctness
- bounding boxes where available
- transformed 224×224 display boxes
- checkpoint used

### 8.1 RSNA Grad-CAM

bash python save_xai_gradcam_results_rsna.py \   --ckpt models/mcc/r50_m.tar \   --test_csv /content/datasets/RSNA_Extracted/csv/rsna_test.csv \   --labels_csv /content/datasets/RSNA_Extracted/csv/stage_2_train_labels.csv \   --out_dir xai_gradcam_results_rsna 

### 8.2 RSNA Integrated Gradients

bash python save_xai_ig_results_rsna.py \   --ckpt models/mcc/r50_m.tar \   --test_csv /content/datasets/RSNA_Extracted/csv/rsna_test.csv \   --labels_csv /content/datasets/RSNA_Extracted/csv/stage_2_train_labels.csv \   --out_dir xai_ig_results_rsna \   --steps 32 

### 8.3 NIH Grad-CAM

Replace the checkpoint path with the NIH fine-tuned checkpoint if needed:

bash python save_xai_gradcam_results_nih.py \   --ckpt outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/model-best.tar \   --test_csv /content/datasets/NIH_Extracted/csv/nih_pneumonia_test.csv \   --out_dir xai_gradcam_results_nih 

### 8.4 NIH Integrated Gradients

bash python save_xai_ig_results_nih.py \   --ckpt outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/model-best.tar \   --test_csv /content/datasets/NIH_Extracted/csv/nih_pneumonia_test.csv \   --out_dir xai_ig_results_nih \   --steps 32 

---

## 9. Recreating XAI visualizations

The saved .npy files can be visualized later without rerunning the model.

Example Grad-CAM visualization:

bash python recreate_xai_from_saved.py \   --metadata xai_gradcam_results_rsna/metadata.csv \   --index 0 \   --out recreated_rsna_gradcam_index0.png 

Example Integrated Gradients visualization:

bash python recreate_ig_from_saved.py \   --metadata xai_ig_results_rsna/metadata.csv \   --index 0 \   --out recreated_rsna_ig_index0.png 

---

## 10. Files intentionally not tracked

The following are intentionally ignored by .gitignore:

text models/ outputs/ eval_outputs/ xai_gradcam_results*/ xai_ig_results*/ *.tar *.pth *.pt *.ckpt *.npy /content/datasets/ data/ 

This keeps the repository lightweight and avoids uploading large medical datasets, checkpoints, and result files.

---

## 11. Original citation

If you use the original CXR-CLIP implementation or pretrained models, cite:

bibtex @incollection{You_2023,     doi = {10.1007/978-3-031-43895-0_10},     url = {https://doi.org/10.1007%2F978-3-031-43895-0_10},     year = 2023,     publisher = {Springer Nature Switzerland},     pages = {101--111},     author = {Kihyun You and Jawook Gu and Jiyeon Ham and Beomhee Park and Jiho Kim and Eun K. Hong and Woonhyuk Baek and Byungseok Roh},     title = {CXR-CLIP: Toward Large Scale Chest X-ray Language-Image Pre-training},     booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023}, } 

---

## 12. License

The original CXR-CLIP project is licensed under CC BY-NC 4.0.

This fork preserves the original license and attribution.

---

## 13. Original contact

Original CXR-CLIP contacts:

Kihyun You: kihyun.you@soombit.ai  
Jawook Gu: jawook.gu@soombit.ai
