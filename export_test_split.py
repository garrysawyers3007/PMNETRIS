"""
export_test_split.py
---------------------
Reads datasetRIS_32x32/metadata.json, reconstructs the flat per-augmentation
record list (mirroring PMnet_data_usc._process_metadata but without the rx_map
key requirement), then applies the same seed and split ratio as train.py to
identify the test subset.

Outputs (written to datasetRIS_32x32/):
  test_split_all.csv            -- every test sample (all augmentation indices)
  test_split_non_rotated.csv    -- only augmentation_index == 0 (original view)

Both CSVs have a `scenario` column: "RIS" or "noRIS".

Augmentation indices:
  0 = original (non-rotated)
  1 = vertical flip
  2 = horizontal flip
  3 = both flipped
"""

import os
import json
import csv
import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset, random_split


# ── match train.py exactly ────────────────────────────────────────────────────
DATA_ROOT   = "datasetRIS_32x32/"
TRAIN_RATIO = 0.9
SEED        = 1234


def set_seed(seed: int) -> None:
    """Exact copy from train.py."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ── helpers ───────────────────────────────────────────────────────────────────

def _xyz_to_rho_theta_phi(x: float, y: float, z: float):
    """Spherical coordinates matching add_polar_coords.py xyz_to_rho_theta_phi."""
    rho   = math.sqrt(x**2 + y**2 + z**2)
    theta = math.atan2(y, x)
    phi   = math.atan2(z, math.sqrt(x**2 + y**2))
    return rho, theta, phi


class _IndexDataset(Dataset):
    """Trivial dataset that returns indices — used solely for random_split."""
    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> int:
        return idx


# ── metadata processing ───────────────────────────────────────────────────────

def build_flat_records(raw_metadata: list) -> list:
    """
    Mirrors PMnet_data_usc._process_metadata with two changes:
      1. rx_map is NOT required in paths_dict (absent in datasetRIS_32x32).
      2. augmentation_index is tracked per data point.
    Spatial fields are stored as raw floats (not normalised tensors).
    """
    flat = []
    for record_idx, record in enumerate(raw_metadata):
        paths_dict = record.get("paths")
        if not isinstance(paths_dict, dict) or \
           not all(k in paths_dict for k in ["city_map", "tx_map", "power_map"]):
            continue

        num_aug = len(paths_dict["city_map"]) if paths_dict.get("city_map") else 0
        if num_aug == 0:
            continue

        for i in range(num_aug):
            if not (len(paths_dict["tx_map"]) > i and len(paths_dict["power_map"]) > i):
                continue

            dp = {
                "tx_id":             record.get("tx_id"),
                "scenario":          record.get("type"),   # "RIS" or "noRIS"
                "augmentation_index": i,
                "is_rotated":        i > 0,
                "city_map_path":     paths_dict["city_map"][i],
                "tx_map_path":       paths_dict["tx_map"][i],
                "power_map_path":    paths_dict["power_map"][i],
                # spatial — filled for RIS, left None for noRIS
                "ris_x":    None, "ris_y":    None, "ris_z":    None,
                "rx_x":     None, "rx_y":     None, "rx_z":     None,
                "ris_roll": None, "ris_pitch": None, "ris_yaw":  None,
                "rho":      None, "theta_rad": None, "phi_rad":  None,
            }

            if record.get("type") == "RIS":
                aug_pos    = record.get("ris_positions_xyz_augmented_for_view")
                aug_rx_pos = record.get("rx_positions_xyz_augmented_for_view")
                aug_orient = record.get("orientations_rpy_augmented_for_view")
                aug_rtp    = record.get("ris_positions_rho_theta_phi_augmented_for_view")

                # RIS XYZ + rho/theta/phi
                if isinstance(aug_pos, list) and len(aug_pos) > i:
                    p = aug_pos[i]
                    if isinstance(p, list) and len(p) == 3:
                        dp["ris_x"], dp["ris_y"], dp["ris_z"] = p
                        # use stored rho/theta/phi if available, else compute from xyz
                        if isinstance(aug_rtp, list) and len(aug_rtp) > i and \
                                isinstance(aug_rtp[i], list) and len(aug_rtp[i]) == 3:
                            dp["rho"], dp["theta_rad"], dp["phi_rad"] = aug_rtp[i]
                        else:
                            dp["rho"], dp["theta_rad"], dp["phi_rad"] = \
                                _xyz_to_rho_theta_phi(*p)

                # RX XYZ
                if isinstance(aug_rx_pos, list) and len(aug_rx_pos) > i:
                    rx = aug_rx_pos[i]
                    if isinstance(rx, list) and len(rx) == 3:
                        dp["rx_x"], dp["rx_y"], dp["rx_z"] = rx

                # RIS orientation (roll, pitch, yaw)
                if isinstance(aug_orient, list) and len(aug_orient) > i:
                    o = aug_orient[i]
                    if isinstance(o, list) and len(o) == 3:
                        dp["ris_roll"], dp["ris_pitch"], dp["ris_yaw"] = o

            flat.append(dp)

    return flat


# ── CSV writer ────────────────────────────────────────────────────────────────

CSV_COLS = [
    "flat_index", "tx_id", "scenario", "augmentation_index", "is_rotated",
    "city_map_path", "tx_map_path", "power_map_path",
    "ris_x", "ris_y", "ris_z",
    "rx_x",  "rx_y",  "rx_z",
    "ris_roll", "ris_pitch", "ris_yaw",
    "rho", "theta_rad", "phi_rad",
]


def write_csv(path: str, rows: list) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows):>6} rows → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)

    metadata_path = os.path.join(DATA_ROOT, "metadata.json")
    print(f"Loading metadata from {metadata_path} ...")
    with open(metadata_path, "r") as f:
        raw_metadata = json.load(f)

    flat_records = build_flat_records(raw_metadata)
    print(f"Total flat records (all augmentations): {len(flat_records)}")

    # Replicate random_split from train.py using the same seed
    train_size = int(len(flat_records) * TRAIN_RATIO)
    test_size  = len(flat_records) - train_size
    _, test_ds = random_split(_IndexDataset(len(flat_records)), [train_size, test_size])
    print(f"Train size: {train_size},  Test size: {test_size}")

    # Build CSV rows for the test subset
    test_rows = []
    for flat_idx in sorted(test_ds.indices):
        dp  = flat_records[flat_idx]
        row = {"flat_index": flat_idx}
        row.update(dp)
        test_rows.append({col: row.get(col, "") for col in CSV_COLS})

    all_csv          = os.path.join(DATA_ROOT, "test_split_all.csv")
    non_rotated_csv  = os.path.join(DATA_ROOT, "test_split_non_rotated.csv")
    non_rotated_rows = [r for r in test_rows if not r["is_rotated"]]

    print("\nWriting CSVs ...")
    write_csv(all_csv,         test_rows)
    write_csv(non_rotated_csv, non_rotated_rows)

    # Quick sanity summary
    ris_count    = sum(1 for r in test_rows if r["scenario"] == "RIS")
    noris_count  = sum(1 for r in test_rows if r["scenario"] == "noRIS")
    print(f"\nTest split summary:")
    print(f"  RIS samples    : {ris_count}")
    print(f"  noRIS samples  : {noris_count}")
    print(f"  Non-rotated    : {len(non_rotated_rows)}")
    print("Done.")


if __name__ == "__main__":
    main()
