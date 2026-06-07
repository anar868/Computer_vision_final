import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import AutoTokenizer

from cxrclip.model import build_model


DEFAULT_CKPT_PATH = "models/mcc/r50_nih.tar"
DEFAULT_TEST_CSV = "/content/datasets/NIH_Extracted/csv/nih_pneumonia_test.csv"
DEFAULT_OUT_DIR = "xai_gradcam_results_nih"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--test_csv", type=str, default=DEFAULT_TEST_CSV)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
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
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_box_transform_params(pil_img):
    png_w, png_h = pil_img.size

    scale_to_resize = 224 / min(png_w, png_h)
    resized_w = int(png_w * scale_to_resize)
    resized_h = int(png_h * scale_to_resize)

    crop_left = max((resized_w - 224) / 2, 0)
    crop_top = max((resized_h - 224) / 2, 0)

    return {
        "png_w": png_w,
        "png_h": png_h,
        "scale_to_resize": scale_to_resize,
        "resized_w": resized_w,
        "resized_h": resized_h,
        "crop_left": crop_left,
        "crop_top": crop_top,
    }


def transform_box_to_display(box, params):
    x = box["x"]
    y = box["y"]
    w = box["width"]
    h = box["height"]

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


def compute_gradcam_for_image(model, cfg, tokenizer, model_name, input_tensor, device):
    activations = {}
    gradients = {}

    target_layer = model.image_encoder.resnet.layer4

    def forward_hook(module, inp, out):
        activations["value"] = out

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)
    info = {}

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

        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_emb = model.text_projection(text_features) if model.projection else text_features
            text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)

        similarities = image_emb @ text_emb.T
        probs = torch.softmax(similarities, dim=1)

        normal_score = similarities[0, 0]
        pneumonia_score = similarities[0, 1]
        target_score = pneumonia_score - normal_score

        info["xai_mode"] = "zero_shot_similarity"
        info["normal_score"] = float(normal_score.detach().cpu())
        info["pneumonia_score"] = float(pneumonia_score.detach().cpu())
        info["normal_probability"] = float(probs[0, 0].detach().cpu())
        info["pneumonia_probability"] = float(probs[0, 1].detach().cpu())
        info["pred_label"] = int(info["pneumonia_probability"] >= 0.5)
        info["pred_class"] = "Pneumonia" if info["pred_label"] == 1 else "Normal"

    else:
        batch = {
            "images": input_tensor,
            "labels": torch.tensor([[1.0]], device=device),
        }

        out = model(batch, device=device)

        logit = out["cls_pred"][0, 0]
        prob = torch.sigmoid(logit)
        target_score = logit

        info["xai_mode"] = "finetuned_classifier"
        info["pneumonia_logit"] = float(logit.detach().cpu())
        info["pneumonia_probability"] = float(prob.detach().cpu())
        info["pred_label"] = int(info["pneumonia_probability"] >= 0.5)
        info["pred_class"] = "Pneumonia" if info["pred_label"] == 1 else "Normal"

    target_score.backward()

    acts = activations["value"]
    grads = gradients["value"]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1)
    cam = F.relu(cam)

    cam = F.interpolate(
        cam.unsqueeze(1),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )

    cam = cam.squeeze().detach().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.astype(np.float32)

    h1.remove()
    h2.remove()

    return cam, info


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    cams_dir = out_dir / "cams"
    out_dir.mkdir(parents=True, exist_ok=True)
    cams_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model, cfg, tokenizer, model_name = load_model(args.ckpt, device)

    test_df = pd.read_csv(args.test_csv).reset_index(drop=True)

    if args.limit is not None:
        test_df = test_df.iloc[args.start: args.start + args.limit].reset_index(drop=True)
    else:
        test_df = test_df.iloc[args.start:].reset_index(drop=True)

    transform = get_preprocess()
    records = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        image_path = row["image"]
        image_id = row["Image Index"] if "Image Index" in row else Path(image_path).name
        patient_id = Path(image_path).stem

        try:
            pil_img = Image.open(image_path).convert("RGB")
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            input_tensor.requires_grad_(True)

            cam, pred_info = compute_gradcam_for_image(
                model=model,
                cfg=cfg,
                tokenizer=tokenizer,
                model_name=model_name,
                input_tensor=input_tensor,
                device=device,
            )

            original_boxes = json.loads(row["boxes_json"]) if "boxes_json" in row else []
            box_params = get_box_transform_params(pil_img)
            display_boxes = [
                transform_box_to_display(box, box_params)
                for box in original_boxes
            ]

            cam_path = cams_dir / f"{patient_id}.npy"
            np.save(cam_path, cam)

            gt_label = int(row["label"]) if "label" in row else None
            pred_label = pred_info["pred_label"]

            record = {
                "patient_id": patient_id,
                "image_id": image_id,
                "image_path": image_path,
                "cam_path": str(cam_path),
                "gt_label": gt_label,
                "gt_class": row["class"] if "class" in row else None,
                "has_bbox": bool(row["has_bbox"]) if "has_bbox" in row else len(original_boxes) > 0,
                "pred_label": pred_label,
                "pred_class": pred_info["pred_class"],
                "correct": int(gt_label == pred_label) if gt_label is not None else None,
                "xai_mode": pred_info["xai_mode"],
                "num_boxes": len(original_boxes),
                "original_boxes_json": json.dumps(original_boxes),
                "display_boxes_json": json.dumps(display_boxes),
                "box_transform_params_json": json.dumps(box_params),
                "checkpoint": args.ckpt,
            }

            for k, v in pred_info.items():
                if k not in record:
                    record[k] = v

            records.append(record)

        except Exception as e:
            print("ERROR:", patient_id, e)
            records.append({
                "patient_id": patient_id,
                "image_id": image_id,
                "image_path": image_path,
                "cam_path": None,
                "gt_label": int(row["label"]) if "label" in row else None,
                "gt_class": row["class"] if "class" in row else None,
                "error": str(e),
                "checkpoint": args.ckpt,
            })

    meta_df = pd.DataFrame(records)
    metadata_csv = out_dir / "metadata.csv"
    meta_df.to_csv(metadata_csv, index=False)

    print("saved metadata:", metadata_csv)
    print("saved CAMs:", cams_dir)

    print("\nSummary:")
    if "correct" in meta_df.columns:
        print("Correct:")
        print(meta_df["correct"].value_counts(dropna=False))
    if "gt_label" in meta_df.columns:
        print("GT labels:")
        print(meta_df["gt_label"].value_counts(dropna=False))
    if "pred_label" in meta_df.columns:
        print("Pred labels:")
        print(meta_df["pred_label"].value_counts(dropna=False))
    if "has_bbox" in meta_df.columns:
        print("Has bbox:")
        print(meta_df["has_bbox"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
