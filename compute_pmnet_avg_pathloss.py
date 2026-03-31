"""
compute_pmnet_avg_pathloss.py
-------------------------------
Compute per-record average path loss from a trained PMNet model on the
GeoRIS dataset (datasetGeoRIS/). Only non-augmented inputs are used
(augmentation index 0 from every metadata record).

De-normalization (from preprocessRIS.py):
  The power maps are stored as uint8 pixels via:
      pixel = clip(path_gain_dB + 255, 0, 255)   FLOOR = -200 dB
  Inverse:
      path_gain_dB = pred_normalized * 255 - 255
      path_gain_dB  = clip(path_gain_dB, FLOOR, 0)

The conditioning vector fed to PMNet (cond_features=7) is the rho/theta/phi tensor
(ris_rho_theta_phi_tensor) from the dataloader:
  [rho/400,             # RIS distance from origin, normalised by 400
   theta,               # azimuth  = atan2(y, x)                 (radians)
   phi,                 # elevation = atan2(z, sqrt(x^2+y^2))   (radians)
   ris_roll,            # RIS orientation (radians)
   ris_pitch,
   ris_yaw,
   is_ris_present_flag] # 1.0 / 0.0

Matches add_polar_coords.py xyz_to_rho_theta_phi and dataloader ris_pos_rho_theta_phi.
"""

import os
import json
import csv
import argparse

import numpy as np
import torch
from torchvision import transforms
from skimage import io
from tqdm import tqdm

from network.pmnet import PMNet

# ── constants ────────────────────────────────────────────────────────────────
FLOOR: float = -200.0          # dB floor used in preprocessRIS.py

IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256), antialias=True),
])

CSV_HEADER = [
    "tx_id", "type", "record_index",
    "ris_x", "ris_y", "ris_z",
    "rx_x",  "rx_y",  "rx_z",
    "avg_pathloss_predicted_dB",
    "avg_pathloss_gt_dB",
    "avg_pixelwise_error_dB",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_gray(path: str) -> np.ndarray:
    """Load an image as a 2-D uint8 array (first channel if RGB)."""
    img = np.asarray(io.imread(path))
    if img.ndim == 3:
        img = img[:, :, 0]
    return img


def pixels_to_db(pixels: np.ndarray) -> np.ndarray:
    """
    Convert uint8 / float pixel values in [0, 255] to path-gain dB in
    [FLOOR, 0].  This reverses the preprocessing in preprocessRIS.py:
        pixel = clip(path_gain_dB + 255, 0, 255)
    """
    db = pixels.astype(np.float32) - 255.0   # [-255, 0]
    return np.clip(db, FLOOR, 0.0)


def avg_pathloss_from_pixels(pixels: np.ndarray) -> float:
    """Average path loss over all map cells (buildings included at FLOOR)."""
    return float(np.mean(pixels_to_db(pixels)))


def avg_pathloss_from_pred(pred_normalized: np.ndarray) -> float:
    """
    pred_normalized: float32 array in [0, 1] (model output after clamp).
    Average path loss in dB.
    """
    return avg_pathloss_from_pixels(pred_normalized * 255.0)


# ── core ──────────────────────────────────────────────────────────────────────

def build_ris_params(record: dict) -> np.ndarray:
    """
    Build the 7-d rho/theta/phi conditioning vector (ris_rho_theta_phi_tensor)
    used by the model:
        [rho/400, theta, phi, ris_roll, ris_pitch, ris_yaw, is_ris_present_flag]

    rtp3 is read directly from record["ris_true_world_pos_rho_theta_phi"]
    (populated by add_polar_coords.py).
    """
    rec_type = record.get("type", "noRIS")

    if rec_type == "RIS":
        rtp3 = record.get("ris_true_world_pos_rho_theta_phi")  # [rho, theta, phi]

        ris_orient = np.array(
            record.get("ris_true_world_orientation_rpy") or
            record["orientations_rpy_augmented_for_view"][0],
            dtype=np.float32,
        )
        params = np.array(
            [rtp3[0] / 400.0] + rtp3[1:] + list(ris_orient) + [1.0],
            dtype=np.float32,
        )
    else:
        params = np.zeros(7, dtype=np.float32)

    return params


def run_inference(
    model: torch.nn.Module,
    metadata: list,
    device: torch.device,
    batch_size: int,
) -> list:
    """
    Iterate over all records, run batched inference, return rows for CSV.
    Only the augmentation-index-0 images are used (non-augmented).
    """
    rows = []
    batch_inputs, batch_params, batch_meta = [], [], []

    def flush_batch():
        if not batch_inputs:
            return

        inputs_t = torch.stack(batch_inputs).to(device)
        params_t = torch.stack(batch_params).to(device)

        with torch.no_grad():
            preds = model(inputs_t, params_t)
            preds = torch.clamp(preds, 0.0, 1.0)

        for pred_t, meta in zip(preds, batch_meta):
            pred_np = pred_t[0].cpu().numpy()          # (H, W)
            pred_db = pixels_to_db(pred_np * 255.0)
            gt_db   = pixels_to_db(meta["gt_pixels"])
            avg_pixelwise_err = float(np.mean(np.abs(pred_db - gt_db)))
            rows.append({
                "tx_id":                    meta["tx_id"],
                "type":                     meta["rec_type"],
                "record_index":             meta["record_index"],
                "ris_x":                    meta["ris_pos"][0] if meta["rec_type"] == "RIS" else "",
                "ris_y":                    meta["ris_pos"][1] if meta["rec_type"] == "RIS" else "",
                "ris_z":                    meta["ris_pos"][2] if meta["rec_type"] == "RIS" else "",
                "rx_x":                     meta["rx_pos"][0]  if meta["rec_type"] == "RIS" else "",
                "rx_y":                     meta["rx_pos"][1]  if meta["rec_type"] == "RIS" else "",
                "rx_z":                     meta["rx_pos"][2]  if meta["rec_type"] == "RIS" else "",
                "avg_pathloss_predicted_dB": avg_pathloss_from_pred(pred_np),
                "avg_pathloss_gt_dB":        meta["avg_gt"],
                "avg_pixelwise_error_dB":    avg_pixelwise_err,
            })

        batch_inputs.clear()
        batch_params.clear()
        batch_meta.clear()

    model.eval()
    for record in tqdm(metadata, desc="Inference"):
        paths = record.get("paths", {})

        # — load non-augmented images (index 0) —
        img_city  = load_gray(paths["city_map"][0])
        img_tx    = load_gray(paths["tx_map"][0])
        img_rx    = load_gray(paths["rx_map"][0])
        img_power = load_gray(paths["power_map"][0])

        # stack: [buildings, TX, RX]  → (3, H, W) float tensor in [0, 1]
        inputs_np = np.stack([img_city, img_tx, img_rx], axis=-1)
        inputs_t  = IMAGE_TRANSFORM(inputs_np).float()

        params_np = build_ris_params(record)
        params_t  = torch.tensor(params_np, dtype=torch.float32)

        rec_type = record.get("type", "noRIS")
        ris_pos  = (record.get("ris_true_world_pos") or [0, 0, 0])
        rx_pos   = (record.get("rx_true_world_pos_for_steering") or [0, 0, 0])

        batch_inputs.append(inputs_t)
        batch_params.append(params_t)
        batch_meta.append({
            "tx_id":        record.get("tx_id"),
            "rec_type":     rec_type,
            "record_index": record.get("record_index"),
            "ris_pos":      ris_pos,
            "rx_pos":       rx_pos,
            "avg_gt":       avg_pathloss_from_pixels(img_power),
            "gt_pixels":    img_power,
        })

        if len(batch_inputs) >= batch_size:
            flush_batch()

    flush_batch()   # remainder
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute avg path loss from PMNet predictions on GeoRIS dataset"
    )
    parser.add_argument(
        "-m", "--model", required=True,
        help="Path to the trained PMNet checkpoint (.pt)",
    )
    parser.add_argument(
        "-d", "--dataset_dir", default="datasetGeoRIS/",
        help="Dataset directory containing metadata.json (default: datasetGeoRIS/)",
    )
    parser.add_argument(
        "-o", "--output_csv", default="pmnet_avg_pathloss.csv",
        help="Output CSV file path (default: pmnet_avg_pathloss.csv)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Inference batch size (default: 16)",
    )
    parser.add_argument(
        "--cond_features", type=int, default=7,
        help="PMNet conditioning-vector size (default: 7 for rho/theta/phi tensor)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── load metadata ─────────────────────────────────────────────────────────
    metadata_path = os.path.join(args.dataset_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    print(f"Records: {len(metadata)} loaded from {metadata_path}")

    # ── load model ────────────────────────────────────────────────────────────
    model = PMNet(
        n_blocks=[3, 3, 27, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=8,
        cond_features=args.cond_features,
    )
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    print(f"Model  : {args.model}")

    # ── inference ─────────────────────────────────────────────────────────────
    rows = run_inference(model, metadata, device, args.batch_size)

    # ── write CSV ─────────────────────────────────────────────────────────────
    print(f"\nWriting {len(rows)} rows to {args.output_csv} …")
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    # ── summary stats ─────────────────────────────────────────────────────────
    pred_vals  = [r["avg_pathloss_predicted_dB"] for r in rows]
    gt_vals    = [r["avg_pathloss_gt_dB"]        for r in rows]
    pw_errors  = [r["avg_pixelwise_error_dB"]    for r in rows]
    print(f"\nSummary ({len(rows)} records):")
    print(f"  Predicted avg pathloss    : {np.mean(pred_vals):.2f} dB  (std {np.std(pred_vals):.2f})")
    print(f"  GT      avg pathloss      : {np.mean(gt_vals):.2f} dB  (std {np.std(gt_vals):.2f})")
    print(f"  Mean avg pixelwise error  : {np.mean(pw_errors):.2f} dB  (std {np.std(pw_errors):.2f})")
    print("Done.")


if __name__ == "__main__":
    main()
