import numpy as np
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, RIS
import os
import json
import csv

# --- Constants ---
ROOT = "DataRIS/"
OUTPUT_CSV = "avg_pathloss_results.csv"
TX_POS_FILE = f"{ROOT}tx_positions.npy"
SCENE_FILE = f"{ROOT}USC_3D/USC.xml"
JSON_INPUT_FILE = "dataset_new_32x32.json"

CM_CELL_SIZE = (5.0, 5.0)  # Cell size in meters
FLOOR = -200.0              # Min path loss value (dB)

# RIS parameters
RIS_NUM_ROWS = 32
RIS_NUM_COLS = 32
RIS_NUM_MODES = 1
RIS_INITIAL_ORIENTATION_EULER = np.array([-np.pi / 2, 0.0, 0.0])  # [roll, pitch, yaw]


def get_scene(xml_file):
    scene = load_scene(xml_file)
    scene.tx_array = PlanarArray(
        num_rows=4, num_cols=4,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V"
    )
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V"
    )
    accepted_mat = ["itu_concrete", "itu_very_dry_ground"]
    for obj_name in scene.objects:
        obj = scene.get(obj_name)
        if obj.radio_material.name not in accepted_mat:
            obj.radio_material = "itu_concrete"
    return scene


def compute_avg_pathloss(scene, cm_center, is_ris=False):
    """
    Runs coverage_map and returns average path loss in dB,
    floored at FLOOR dB (ignoring -inf / no-coverage cells).
    """
    coverage_map_params = {
        "cm_cell_size": CM_CELL_SIZE,
        "cm_center": cm_center,
        "cm_size": [50, 50],
        "cm_orientation": [0, 0, 0],
        "diffraction": True,
        "scattering": True,
        "edge_diffraction": True,
        "max_depth": 1000,
        "num_samples": int(2e6),
    }
    if is_ris:
        coverage_map_params["ris"] = True

    cm = scene.coverage_map(**coverage_map_params)

    if cm.path_gain.shape[0] == 0:
        return float("nan")

    path_gain_db = 10.0 * np.log10(cm.path_gain[0].numpy())

    # Replace -inf (zero path gain) with the floor value
    path_gain_db[path_gain_db == -np.inf] = FLOOR
    # Clamp positive gains to 0 (shouldn't happen physically but guard anyway)
    path_gain_db[path_gain_db > 0] = 0.0
    # Apply floor
    path_gain_db[path_gain_db < FLOOR] = FLOOR

    avg_pl = float(np.mean(path_gain_db))
    return avg_pl


def main():
    print("Loading input files...")
    with open(JSON_INPUT_FILE, "r") as f:
        data = json.load(f)
    tx_positions_all = np.load(TX_POS_FILE)
    print(f"Loaded {len(data)} TX entries.")

    csv_rows = []
    csv_header = [
        "tx_id",
        "scenario",          # "RIS" or "noRIS"
        "tx_x", "tx_y", "tx_z",
        "ris_x", "ris_y", "ris_z",
        "rx_x", "rx_y", "rx_z",
        "avg_pathloss_dB",
    ]

    for tx_entry in data:
        tx_id_str = tx_entry["tx_id"]
        print(f"\nProcessing TX_ID: {tx_id_str}")

        try:
            tx_id_int = int(tx_id_str)
        except ValueError:
            print(f"  Warning: invalid TX_ID '{tx_id_str}', skipping.")
            continue

        if not (0 <= tx_id_int < len(tx_positions_all)):
            print(f"  Warning: TX_ID {tx_id_int} out of bounds, skipping.")
            continue

        tx_pos = tx_positions_all[tx_id_int]
        tx_pos_3d = np.array([tx_pos[0], tx_pos[1], tx_pos[2]])
        tx_pos_scene = tx_pos_3d + np.array([0.0, 0.0, 2.0])  # antenna height offset

        # ----------------------------------------------------------------
        # No-RIS cases — one map per unique RX position
        # ----------------------------------------------------------------
        unique_rx_positions = {}
        for record in tx_entry["records"]:
            rx_key = tuple(record["RX"])
            if rx_key not in unique_rx_positions:
                unique_rx_positions[rx_key] = record["RX"]

        for rx_idx, rx_pos_list in enumerate(unique_rx_positions.values()):
            print(f"  noRIS case {rx_idx}: RX={rx_pos_list}")
            scene = get_scene(SCENE_FILE)
            tx_obj = Transmitter(f"tx{tx_id_str}_noris_{rx_idx}", tx_pos_scene, [0.0, 0.0, 0.0])
            scene.add(tx_obj)

            avg_pl = compute_avg_pathloss(scene, cm_center=rx_pos_list, is_ris=False)
            print(f"    avg pathloss = {avg_pl:.2f} dB")

            csv_rows.append({
                "tx_id": tx_id_str,
                "scenario": "noRIS",
                "tx_x": tx_pos_3d[0], "tx_y": tx_pos_3d[1], "tx_z": tx_pos_3d[2],
                "ris_x": "", "ris_y": "", "ris_z": "",
                "rx_x": rx_pos_list[0], "rx_y": rx_pos_list[1], "rx_z": rx_pos_list[2],
                "avg_pathloss_dB": avg_pl,
            })

        # ----------------------------------------------------------------
        # RIS cases
        # ----------------------------------------------------------------
        print(f"  Processing {len(tx_entry['records'])} RIS cases...")
        for i, record in enumerate(tx_entry["records"]):
            ris_pos = np.array(record["RIS"])
            rx_pos = np.array(record["RX"])
            print(f"  RIS case {i}: RIS={ris_pos.tolist()}, RX={rx_pos.tolist()}")

            scene = get_scene(SCENE_FILE)
            tx_obj = Transmitter(f"tx{tx_id_str}_ris_{i}", tx_pos_scene, [0.0, 0.0, 0.0])
            scene.add(tx_obj)

            rx_obj = Receiver(f"rx_steer_{tx_id_str}_{i}", rx_pos)
            scene.add(rx_obj)

            ris_obj = RIS(
                name=f"ris_{tx_id_str}_{i}",
                position=ris_pos,
                orientation=RIS_INITIAL_ORIENTATION_EULER,
                num_rows=RIS_NUM_ROWS,
                num_cols=RIS_NUM_COLS,
                num_modes=RIS_NUM_MODES,
            )
            ris_obj.look_at((tx_obj.position + rx_obj.position) / 2.0)
            scene.add(ris_obj)
            ris_obj.phase_gradient_reflector(tx_obj.position, rx_obj.position)

            avg_pl = compute_avg_pathloss(scene, cm_center=rx_pos.tolist(), is_ris=True)
            print(f"    avg pathloss = {avg_pl:.2f} dB")

            csv_rows.append({
                "tx_id": tx_id_str,
                "scenario": "RIS",
                "tx_x": tx_pos_3d[0], "tx_y": tx_pos_3d[1], "tx_z": tx_pos_3d[2],
                "ris_x": ris_pos[0], "ris_y": ris_pos[1], "ris_z": ris_pos[2],
                "rx_x": rx_pos[0], "rx_y": rx_pos[1], "rx_z": rx_pos[2],
                "avg_pathloss_dB": avg_pl,
            })

    # ----------------------------------------------------------------
    # Write CSV
    # ----------------------------------------------------------------
    print(f"\nWriting {len(csv_rows)} rows to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("Done.")


if __name__ == "__main__":
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {gpus if gpus else 'None (CPU mode)'}")

    main()
