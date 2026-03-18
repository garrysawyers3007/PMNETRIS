import os
import numpy as np
import torch
from torchvision import transforms
from skimage import io
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from network.pmnet import PMNetFiLM

# --- Paths ---
CSV_INPUT_PATH = "datasetNegative/metadata.csv"
CITY_MAP_PATH = "DataRIS/USC_city_map.png"
MODEL_PATH = "datasetRISNewCorrected/PMNet_results/augmented_config_USC_pmnetV3_V2_epoch100/8_0.0001_0.45_10/model_0.03808.pt"

# --- Coordinate normalisation bounds (must match training) ---
RIS_POS_MIN = np.array([-400.0, -400.0, 0.0], dtype=np.float32)
RIS_POS_MAX = np.array([400.0,  400.0, 55.0], dtype=np.float32)

# Pixel scale factor: dB = norm * 255 - 255  →  MAE_dB = MAE_norm * 255
DB_SCALE = 255.0


def normalise(world_xyz, pos_min, pos_max):
    """Normalise world coordinates to [0, 1] using the given bounds."""
    return (np.array(world_xyz, dtype=np.float32) - pos_min) / (pos_max - pos_min)


def evaluate_negative(model, csv_path, city_map_path, ris_pos_min, ris_pos_max, device):
    model.eval()

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV not found at '{csv_path}'.")
        return

    # Load shared city map (same scene for all entries)
    city_map_raw = io.imread(city_map_path)
    if city_map_raw.ndim == 3:
        city_map_raw = city_map_raw[:, :, 0]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
    ])

    results = []

    with torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            try:
                tx_map_raw = io.imread(row["tx_map_path"])
                if tx_map_raw.ndim == 3:
                    tx_map_raw = tx_map_raw[:, :, 0]

                rx_map_raw = io.imread(row["rx_map_path"])
                if rx_map_raw.ndim == 3:
                    rx_map_raw = rx_map_raw[:, :, 0]

                power_map_gt_raw = io.imread(row["power_map_path"])
                if power_map_gt_raw.ndim == 3:
                    power_map_gt_raw = power_map_gt_raw[:, :, 0]
            except FileNotFoundError as e:
                print(f"\nWarning: Missing file {e.filename} at row {index}. Skipping.")
                continue

            # Build 3-channel input [city_map, tx_map, rx_map]
            input_np = np.stack([city_map_raw, tx_map_raw, rx_map_raw], axis=-1)
            inputs_tensor = transform(input_np).float().unsqueeze(0).to(device)
            power_gt_tensor = transforms.ToTensor()(power_map_gt_raw).float().unsqueeze(0).to(device)

            # Normalise RIS and RX world coordinates to [0, 1]
            ris_world = np.array([row["ris_x"], row["ris_y"], row["ris_z"]], dtype=np.float32)
            rx_world  = np.array([row["rx_x"],  row["rx_y"],  row["rx_z"]],  dtype=np.float32)

            ris_norm = normalise(ris_world, ris_pos_min, ris_pos_max)
            rx_norm  = normalise(rx_world,  ris_pos_min, ris_pos_max)

            ris_roll  = float(row["ris_roll"])
            ris_pitch = float(row["ris_pitch"])
            ris_yaw   = float(row["ris_yaw"])
            ris_flag = 1.0  # all entries are RIS scenarios

            # Match sign convention used in qual_from_csv.py: negate RIS y
            ris_params = [
                ris_norm[0], -ris_norm[1], ris_norm[2],
                ris_roll, ris_pitch, ris_yaw,
                rx_norm[0],  rx_norm[1],  rx_norm[2],
                ris_flag,
            ]
            ris_params_tensor = torch.tensor(ris_params, dtype=torch.float32).unsqueeze(0).to(device)

            pred = model(inputs_tensor, ris_params_tensor)
            pred = torch.clamp(pred, 0.0, 1.0)

            # MAE in normalised [0,1] space
            mae_norm = float(torch.mean(torch.abs(pred - power_gt_tensor)).item())
            # MAE in dB space (offset cancels, only scale matters)
            mae_db = mae_norm * DB_SCALE

            results.append({
                "index": index,
                "tx_id": row["tx_id"],
                "mae": mae_norm,
                "mae_db": mae_db,
            })

            print(f"  [tx_id={row['tx_id']}]  MAE={mae_norm:.5f}  MAE(dB)={mae_db:.3f}")

    if not results:
        print("No results computed.")
        return

    all_mae    = [r["mae"]    for r in results]
    all_mae_db = [r["mae_db"] for r in results]

    print("\n" + "=" * 50)
    print(f"Evaluated {len(results)} entries.")
    print(f"Mean MAE        : {np.mean(all_mae):.5f}")
    print(f"Mean MAE (dB)   : {np.mean(all_mae_db):.3f}")
    print(f"Median MAE (dB) : {np.median(all_mae_db):.3f}")
    print(f"Std MAE (dB)    : {np.std(all_mae_db):.3f}")
    print("=" * 50)

    # --- Boxplot ---
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.boxplot(all_mae_db, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.7))
    ax.set_xticks([1])
    ax.set_xticklabels(["Negative samples"])
    ax.set_ylabel("MAE (dB)")
    ax.set_title("MAE (dB) distribution — negative samples")
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(csv_path), "mae_db_boxplot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Boxplot saved to: {plot_path}")

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PMNetFiLM(
        n_blocks=[3, 3, 27, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=8,
        cond_features=10,
    ).to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model weights not found at '{MODEL_PATH}'. Using uninitialised model.")
    else:
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"Loaded model weights from '{MODEL_PATH}'.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            exit(1)

    evaluate_negative(
        model=model,
        csv_path=CSV_INPUT_PATH,
        city_map_path=CITY_MAP_PATH,
        ris_pos_min=RIS_POS_MIN,
        ris_pos_max=RIS_POS_MAX,
        device=device,
    )
