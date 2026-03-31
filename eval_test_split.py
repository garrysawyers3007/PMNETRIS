"""
eval_test_split.py
-------------------
Evaluates a trained PMNet model on the test split of a dataset.
Uses PMnet_data_usc directly, with the same seed and split ratio as train.py.

Outputs (written to <output_dir>/):
  test_eval_all.csv            -- all test samples (all augmentation indices)
  test_eval_non_rotated.csv    -- augmentation_index == 0 only (original view)

CSV columns:
  flat_index, tx_id, scenario, augmentation_index, is_rotated,
  avg_pred_pathloss_dB, avg_gt_pathloss_dB, avg_pixelwise_error_dB

De-normalisation (matches preprocessRIS.py / compute_pmnet_avg_pathloss.py):
  path_gain_dB = clip(normalised_value * 255 - 255, -200, 0)

Conditioning vector (ris_polar_tensor, 7-dim):
  [rho/400, theta, phi, ris_roll, ris_pitch, ris_yaw, is_ris_present_flag]

Usage:
  python eval_test_split.py --model path/to/model.pt [--data_root datasetGeoRIS/]
"""

import os
import csv
import re
import random
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataloader import PMnet_data_usc
from network.pmnet import PMNet


# ── constants (match train.py) ────────────────────────────────────────────────
SEED        = 1234
TRAIN_RATIO = 0.9
RIS_POS_MIN = [-400, -400, 0]
RIS_POS_MAX = [400,  400, 55]
FLOOR_DB    = -200.0

CSV_COLS = [
    "flat_index", "tx_id", "scenario", "augmentation_index", "is_rotated",
    "avg_pred_pathloss_dB", "avg_gt_pathloss_dB", "avg_pixelwise_error_dB",
]

_AUG_RE = re.compile(r"_city_map_(\d+)\.\w+$")


# ── helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Exact copy from train.py."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def pixels_to_db(t: torch.Tensor) -> torch.Tensor:
    """
    Invert preprocessRIS.py normalisation:
        pixel = clip(path_gain_dB + 255, 0, 255)
    Inverse: path_gain_dB = clip(pixel_value * 255 - 255, FLOOR, 0)
    t is a float tensor in [0, 1].
    """
    return torch.clamp(t * 255.0 - 255.0, FLOOR_DB, 0.0)


def parse_aug_index(city_map_path: str) -> int:
    m = _AUG_RE.search(city_map_path)
    return int(m.group(1)) if m else 0


def build_path_lookup(metadata_records: list) -> dict:
    """Maps city_map_path -> {flat_index, tx_id, scenario, augmentation_index, is_rotated}."""
    lut = {}
    for flat_idx, dp in enumerate(metadata_records):
        aug = parse_aug_index(dp["city_map_path"])
        lut[dp["city_map_path"]] = {
            "flat_index":         flat_idx,
            "tx_id":              dp.get("tx_id", ""),
            "scenario":           dp.get("type", ""),
            "augmentation_index": aug,
            "is_rotated":         aug > 0,
        }
    return lut


def write_csv(path: str, rows: list) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows):>6} rows -> {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PMNet on the test split and write per-sample pathloss CSVs."
    )
    parser.add_argument("-m", "--model",       required=True,
                        help="Path to trained PMNet checkpoint (.pt)")
    parser.add_argument("-d", "--data_root",   default="datasetGeoRIS/",
                        help="Dataset directory containing metadata.json")
    parser.add_argument("-o", "--output_dir",  default=None,
                        help="Output directory (defaults to data_root)")
    parser.add_argument("--batch_size",        type=int,   default=8)
    parser.add_argument("--train_ratio",       type=float, default=TRAIN_RATIO)
    parser.add_argument("--cond_features",     type=int,   default=7,
                        help="PMNet conditioning vector size (default: 7 for rho/theta/phi)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_root
    os.makedirs(output_dir, exist_ok=True)

    set_seed(SEED)

    # ── dataset + split (mirrors train.py exactly) ────────────────────────────
    print(f"Loading dataset from {args.data_root} ...")
    full_dataset = PMnet_data_usc(
        dir_dataset=args.data_root,
        ris_pos_min=RIS_POS_MIN,
        ris_pos_max=RIS_POS_MAX,
        get_paths=True,
    )
    print(f"Total samples: {len(full_dataset)}")

    train_size = int(len(full_dataset) * args.train_ratio)
    test_size  = len(full_dataset) - train_size
    _, test_ds = random_split(full_dataset, [train_size, test_size])
    print(f"Train: {train_size}   Test: {test_size}")

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # city_map_path -> {flat_index, tx_id, scenario, augmentation_index, is_rotated}
    path_lut = build_path_lookup(full_dataset.metadata_records)

    # ── model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PMNet(
        n_blocks=[3, 3, 27, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=8,
        cond_features=args.cond_features,
    )
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {args.model}   device: {device}")

    # ── inference ─────────────────────────────────────────────────────────────
    # Dataloader with get_paths=True yields 9-tuple:
    #   inputs, ris_info, ris_polar_tensor (9-dim), ris_rho_theta_phi_tensor (7-dim),
    #   targets, noris_targets, city_paths, tx_paths, power_paths
    #
    # Conditioning: ris_rho_theta_phi_tensor (index 3) -- 7-dim [rho/400, theta, phi,
    # roll, pitch, yaw, flag]. Matches cluster training code and
    # compute_pmnet_avg_pathloss.py build_ris_params().
    rows = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            (inputs, ris_info, ris_polar_tensor, ris_rho_theta_phi_tensor, targets,
             noris_targets, city_paths, tx_paths, power_paths) = batch

            inputs                = inputs.to(device)
            ris_rho_theta_phi_tensor = ris_rho_theta_phi_tensor.to(device)
            targets               = targets.to(device)

            preds = model(inputs, ris_rho_theta_phi_tensor)
            preds = torch.clamp(preds, 0.0, 1.0)

            pred_db = pixels_to_db(preds.squeeze(1))    # [B, H, W]
            gt_db   = pixels_to_db(targets.squeeze(1))  # [B, H, W]

            for i in range(inputs.shape[0]):
                meta = path_lut.get(city_paths[i], {})
                rows.append({
                    "flat_index":             meta.get("flat_index",         ""),
                    "tx_id":                  meta.get("tx_id",              ""),
                    "scenario":               meta.get("scenario",           ""),
                    "augmentation_index":     meta.get("augmentation_index", ""),
                    "is_rotated":             meta.get("is_rotated",         ""),
                    "avg_pred_pathloss_dB":   round(pred_db[i].mean().item(), 4),
                    "avg_gt_pathloss_dB":     round(gt_db[i].mean().item(),   4),
                    "avg_pixelwise_error_dB": round(
                        (pred_db[i] - gt_db[i]).abs().mean().item(), 4
                    ),
                })

    # ── write CSVs ────────────────────────────────────────────────────────────
    all_csv     = os.path.join(output_dir, "test_eval_all.csv")
    nonrot_csv  = os.path.join(output_dir, "test_eval_non_rotated.csv")
    non_rotated = [r for r in rows if not r["is_rotated"]]

    print("\nWriting CSVs ...")
    write_csv(all_csv,    rows)
    write_csv(nonrot_csv, non_rotated)

    # ── summary ───────────────────────────────────────────────────────────────
    ris_count   = sum(1 for r in rows if r["scenario"] == "RIS")
    noris_count = sum(1 for r in rows if r["scenario"] == "noRIS")
    pw_errors   = [r["avg_pixelwise_error_dB"] for r in rows]
    print(f"\nSummary:")
    print(f"  RIS samples    : {ris_count}")
    print(f"  noRIS samples  : {noris_count}")
    print(f"  Non-rotated    : {len(non_rotated)}")
    if pw_errors:
        print(f"  Mean pixelwise error : {np.mean(pw_errors):.2f} dB"
              f"  (std {np.std(pw_errors):.2f})")
    print("Done.")


if __name__ == "__main__":
    main()
