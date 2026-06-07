import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import AutoTokenizer
from captum.attr import IntegratedGradients

from cxrclip.model import build_model


DEFAULT_CKPT_PATH = "models/mcc/r50_m.tar"
DEFAULT_TEST_CSV = "/content/datasets/RSNA_Extracted/csv/rsna_test.csv"
DEFAULT_LABELS_CSV = "/content/datasets/RSNA_Extracted/csv/stage_2_train_labels.csv"
DEFAULT_OUT_DIR = "xai_ig_results"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--test_csv", type=str, default=DEFAULT_TEST_CSV)
    parser.add_argument("--labels_csv", type=str, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=32)
    return parser.parse_args()


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]

    model_name = cfg["model"]["name"]
    print("checkpoint:", ckpt_path)
    print("model name:", model_name)

    if model_name == "clip_custom":
        tokenizer_name = cfg["tokenizer"]["pretrained_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = None

    model = build_model(
        model_config=cfg["model"],
        loss_config=cfg["loss"],
        tokenizer=tokenizer,
    )

    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)
    model.eval()

    return model, cfg, tokenizer, model_name


def get_preprocess():
    model_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return model_transform


def get_boxes_for_patient(labels_df, patient_id):
    boxes = labels_df[
        (labels_df["patientId"] == patient_id) &
        (labels_df["Target"] == 1)
    ].copy()

    original_boxes = []
    for _, b in boxes.iterrows():
        original_boxes.append({
            "x": float(b["x"]),
            "y": float(b["y"]),
            "width": float(b["width"]),
            "height": float(b["height"]),
        })

    return original_boxes


def get_box_transform_params(pil_img):
    orig_dicom_size = 1024
    png_w, png_h = pil_img.size

    scale_from_dicom_to_png_x = png_w / orig_dicom_size
    scale_from_dicom_to_png_y = png_h / orig_dicom_size

    scale_to_resize = 224 / min(png_w, png_h)
    resized_w = int(png_w * scale_to_resize)
    resized_h = int(png_h * scale_to_resize)

    crop_left = max((resized_w - 224) / 2, 0)
    crop_top = max((resized_h - 224) / 2, 0)

    return {
        "orig_dicom_size": orig_dicom_size,
        "png_w": png_w,
        "png_h": png_h,
        "scale_from_dicom_to_png_x": scale_from_dicom_to_png_x,
        "scale_from_dicom_to_png_y": scale_from_dicom_to_png_y,
        "scale_to_resize": scale_to_resize,
        "resized_w": resized_w,
        "resized_h": resized_h,
        "crop_left": crop_left,
        "crop_top": crop_top,
    }


def transform_box_to_display(box, params):
    x = box["x"] * params["scale_from_dicom_to_png_x"]
    y = box["y"] * params["scale_from_dicom_to_png_y"]
    w = box["width"] * params["scale_from_dicom_to_png_x"]
    h = box["height"] * params["scale_from_dicom_to_png_y"]

    x1 = x * params["scale_to_resize"] - params["crop_left"]
    y1 = y * params["scale_to_resize"] - params["crop_top"]
    x2 = (x + w) * params["scale_to_resize"] - params["crop_left"]
    y2 = (y + h) * params["scale_to_resize"] - params["crop_top"]

    x1 = max(0, min(224, x1))
    y1 = max(0, min(224, y1))
    x2 = max(0, min(224, x2))
    y2 = max(0, min(224, y2))

    return {
        "x": float(x1),
        "y": float(y1),
        "width": float(x2 - x1),
        "height": float(y2 - y1),
    }


class IGWrapper(torch.nn.Module):
    def __init__(self, model, cfg, tokenizer, model_name, device):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device

        if model_name == "clip_custom":
            prompts = [
                "No evidence of pneumonia",
                "Findings suggesting pneumonia.",
            ]
            text_tokens = tokenizer(
                prompts,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                max_length=cfg["base"]["text_max_length"],
            ).to(device)

            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_emb = model.text_projection(text_features) if model.projection else text_features
                text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)

            self.register_buffer("text_emb", text_emb)

    def forward(self, images):
        if self.model_name == "clip_custom":
            image_features = self.model.encode_image(images)
            image_emb = self.model.image_projection(image_features) if self.model.projection else image_features
            image_emb = image_emb / image_emb.norm(dim=1, keepdim=True)

            similarities = image_emb @ self.text_emb.T
            normal_score = similarities[:, 0]
            pneumonia_score = similarities[:, 1]

            # Same target as Grad-CAM:
            # pneumonia similarity minus normal similarity
            return pneumonia_score - normal_score

        else:
            batch = {
                "images": images,
                "labels": torch.ones((images.shape[0], 1), device=images.device),
            }
            out = self.model(batch, device=self.device)
            logit = out["cls_pred"][:, 0]

            # Fine-tuned classifier target:
            # pneumonia logit
            return logit


def compute_prediction(model, cfg, tokenizer, model_name, input_tensor, device):
    with torch.no_grad():
        if model_name == "clip_custom":
            prompts = [
                "No evidence of pneumonia",
                "Findings suggesting pneumonia.",
            ]

            text_tokens = tokenizer(
                prompts,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                max_length=cfg["base"]["text_max_length"],
            ).to(device)

            image_features = model.encode_image(input_tensor)
            image_emb = model.image_projection(image_features) if model.projection else image_features
            image_emb = image_emb / image_emb.norm(dim=1, keepdim=True)

            text_features = model.encode_text(text_tokens)
            text_emb = model.text_projection(text_features) if model.projection else text_features
            text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)

            similarities = image_emb @ text_emb.T
            probs = torch.softmax(similarities, dim=1)

            normal_score = float(similarities[0, 0].cpu())
            pneumonia_score = float(similarities[0, 1].cpu())
            pneumonia_probability = float(probs[0, 1].cpu())

            return {
                "xai_mode": "zero_shot_similarity",
                "normal_score": normal_score,
                "pneumonia_score": pneumonia_score,
                "pneumonia_probability": pneumonia_probability,
                "pred_label": int(pneumonia_probability >= 0.5),
                "pred_class": "Pneumonia" if pneumonia_probability >= 0.5 else "Normal",
            }

        else:
            batch = {
                "images": input_tensor,
                "labels": torch.ones((1, 1), device=device),
            }
            out = model(batch, device=device)

            logit = out["cls_pred"][0, 0]
            prob = torch.sigmoid(logit)

            pneumonia_probability = float(prob.cpu())
            return {
                "xai_mode": "finetuned_classifier",
                "pneumonia_logit": float(logit.cpu()),
                "pneumonia_probability": pneumonia_probability,
                "pred_label": int(pneumonia_probability >= 0.5),
                "pred_class": "Pneumonia" if pneumonia_probability >= 0.5 else "Normal",
            }


def make_ig_map(wrapper, input_tensor, steps):
    baseline = torch.zeros_like(input_tensor)

    ig = IntegratedGradients(wrapper)

    attributions = ig.attribute(
        input_tensor,
        baselines=baseline,
        n_steps=steps,
    )

    # attributions shape: [1, 3, 224, 224]
    attr = attributions.detach().cpu().numpy()[0]

    # Convert channel attribution to one 2D heatmap
    attr = np.abs(attr).sum(axis=0)

    attr = attr - attr.min()
    attr = attr / (attr.max() + 1e-8)
    attr = attr.astype(np.float32)

    return attr


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    ig_dir = out_dir / "ig_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    ig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model, cfg, tokenizer, model_name = load_model(args.ckpt, device)
    wrapper = IGWrapper(model, cfg, tokenizer, model_name, device).to(device)
    wrapper.eval()

    test_df = pd.read_csv(args.test_csv).reset_index(drop=True)
    labels_df = pd.read_csv(args.labels_csv)

    if args.limit is not None:
        test_df = test_df.iloc[args.start: args.start + args.limit].reset_index(drop=True)
    else:
        test_df = test_df.iloc[args.start:].reset_index(drop=True)

    transform = get_preprocess()

    records = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        image_path = row["image"]
        patient_id = Path(image_path).stem

        try:
            pil_img = Image.open(image_path).convert("RGB")
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            input_tensor.requires_grad_(True)

            pred_info = compute_prediction(
                model=model,
                cfg=cfg,
                tokenizer=tokenizer,
                model_name=model_name,
                input_tensor=input_tensor,
                device=device,
            )

            ig_map = make_ig_map(wrapper, input_tensor, args.steps)

            original_boxes = get_boxes_for_patient(labels_df, patient_id)
            box_params = get_box_transform_params(pil_img)
            display_boxes = [
                transform_box_to_display(box, box_params)
                for box in original_boxes
            ]

            ig_path = ig_dir / f"{patient_id}.npy"
            np.save(ig_path, ig_map)

            gt_label = int(row["label"]) if "label" in row else None
            pred_label = pred_info["pred_label"]

            record = {
                "patient_id": patient_id,
                "image_path": image_path,
                "ig_path": str(ig_path),
                "gt_label": gt_label,
                "gt_class": row["class"] if "class" in row else None,
                "pred_label": pred_label,
                "pred_class": pred_info["pred_class"],
                "correct": int(gt_label == pred_label) if gt_label is not None else None,
                "xai_mode": pred_info["xai_mode"],
                "num_boxes": len(original_boxes),
                "original_boxes_json": json.dumps(original_boxes),
                "display_boxes_json": json.dumps(display_boxes),
                "box_transform_params_json": json.dumps(box_params),
                "checkpoint": args.ckpt,
                "ig_steps": args.steps,
            }

            for k, v in pred_info.items():
                if k not in record:
                    record[k] = v

            records.append(record)

        except Exception as e:
            print("ERROR:", patient_id, e)
            records.append({
                "patient_id": patient_id,
                "image_path": image_path,
                "ig_path": None,
                "gt_label": int(row["label"]) if "label" in row else None,
                "gt_class": row["class"] if "class" in row else None,
                "error": str(e),
                "checkpoint": args.ckpt,
            })

    meta_df = pd.DataFrame(records)

    metadata_csv = out_dir / "metadata.csv"
    meta_df.to_csv(metadata_csv, index=False)

    print("saved metadata:", metadata_csv)
    print("saved IG maps:", ig_dir)
    print()
    print("Summary:")
    if "correct" in meta_df.columns:
        print(meta_df["correct"].value_counts(dropna=False))
    if "gt_label" in meta_df.columns:
        print("GT labels:")
        print(meta_df["gt_label"].value_counts(dropna=False))
    if "pred_label" in meta_df.columns:
        print("Pred labels:")
        print(meta_df["pred_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
