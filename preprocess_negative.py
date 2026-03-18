import numpy as np
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, RIS
import cv2
import os
import json
import csv

# --- Constants ---
SCENE_FILE = "DataRIS/USC_3D/USC.xml"
JSON_INPUT_FILE = "RX_points_negative.json"
OUTPUT_ROOT = "datasetNegative/"
CSV_OUTPUT_FILE = os.path.join(OUTPUT_ROOT, "metadata.csv")

CM_CELL_SIZE = (5.0, 5.0)
FLOOR = -200  # Min power value (dB)
CM_DIM = 900   # Marker map canvas size (pixels)
TX_WIDTH = 8   # Marker square width in pixels

RIS_NUM_ROWS = 32
RIS_NUM_COLS = 32
RIS_NUM_MODES = 1
RIS_INITIAL_ORIENTATION = np.array([-np.pi / 2, 0.0, 0.0])


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


def make_marker_map(pos_3d):
    """Create a CM_DIM x CM_DIM uint8 image with a white square at the given world (x,y) position."""
    pos_2d = (np.array(pos_3d[:2]) + (CM_DIM // 2) + 50).astype(np.int16)
    img = np.zeros((CM_DIM, CM_DIM), dtype=np.uint8)
    shift = TX_WIDTH // 2
    map_y = CM_DIM - pos_2d[1]
    y0 = int(np.clip(map_y - shift, 0, CM_DIM))
    y1 = int(np.clip(map_y + shift + 1, 0, CM_DIM))
    x0 = int(np.clip(pos_2d[0] - shift, 0, CM_DIM))
    x1 = int(np.clip(pos_2d[0] + shift + 1, 0, CM_DIM))
    img[y0:y1, x0:x1] = 255
    return img


def process_power_map(path_gain_tensor):
    """Convert raw path gain tensor to a uint8 image."""
    cm = 10.0 * np.log10(path_gain_tensor.numpy())
    cm[cm == -np.inf] = -255
    cm[cm > 0] = 0
    cm = cv2.flip(cm, 0)
    cm[cm < FLOOR] = FLOOR
    cm += 255
    return np.clip(cm.astype(np.uint8), 0, 255)


def generate_ris_power_map(tx_pos, ris_pos, rx_pos):
    """Set up scene with TX, RIS, RX and return the processed power map."""
    tx_pos = np.array(tx_pos)
    ris_pos = np.array(ris_pos)
    rx_pos = np.array(rx_pos)

    scene = get_scene(SCENE_FILE)

    tx = Transmitter("tx", tx_pos + [0.0, 0.0, 2.0], [0.0, 0.0, 0.0])
    scene.add(tx)

    rx = Receiver("rx", position=rx_pos)
    scene.add(rx)

    ris = RIS(
        name="ris",
        position=ris_pos,
        orientation=RIS_INITIAL_ORIENTATION,
        num_rows=RIS_NUM_ROWS,
        num_cols=RIS_NUM_COLS,
        num_modes=RIS_NUM_MODES,
    )
    ris.look_at((tx.position + rx.position) / 2.0)
    scene.add(ris)
    ris.phase_gradient_reflector(tx.position, rx.position)

    cm = scene.coverage_map(
        cm_cell_size=CM_CELL_SIZE,
        ris=True,
        cm_center=rx_pos,
        cm_size=[50, 50],
        cm_orientation=[0, 0, 0],
        diffraction=True,
        scattering=True,
        edge_diffraction=True,
        max_depth=1000,
        num_samples=int(2e6),
    )

    orientation = ris.orientation.numpy().tolist()
    return process_power_map(cm.path_gain[0]), orientation


def main():
    with open(JSON_INPUT_FILE, "r") as f:
        entries = json.load(f)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    csv_rows = []

    for entry in entries:
        tx_id = entry["tx_id"]
        tx_pos = entry["tx_pos"]
        ris_pos = entry["ris_pos"]
        rx_pos = entry["rx_pos"]

        print(f"Processing TX_ID {tx_id} ...")
        print(f"  TX:  {tx_pos}")
        print(f"  RIS: {ris_pos}")
        print(f"  RX:  {rx_pos}")

        power_map, ris_orientation = generate_ris_power_map(tx_pos, ris_pos, rx_pos)
        tx_map = make_marker_map(tx_pos)
        rx_map = make_marker_map(rx_pos)

        power_path = os.path.join(OUTPUT_ROOT, f"tx{tx_id}_power_map.png")
        tx_map_path = os.path.join(OUTPUT_ROOT, f"tx{tx_id}_tx_map.png")
        rx_map_path = os.path.join(OUTPUT_ROOT, f"tx{tx_id}_rx_map.png")

        cv2.imwrite(power_path, power_map)
        cv2.imwrite(tx_map_path, tx_map)
        cv2.imwrite(rx_map_path, rx_map)
        print(f"  Saved: {power_path}, {tx_map_path}, {rx_map_path}")

        csv_rows.append({
            "tx_id": tx_id,
            "tx_x": tx_pos[0], "tx_y": tx_pos[1], "tx_z": tx_pos[2],
            "ris_x": ris_pos[0], "ris_y": ris_pos[1], "ris_z": ris_pos[2],
            "ris_roll": ris_orientation[0], "ris_pitch": ris_orientation[1], "ris_yaw": ris_orientation[2],
            "rx_x": rx_pos[0], "rx_y": rx_pos[1], "rx_z": rx_pos[2],
            "power_map_path": os.path.abspath(power_path),
            "tx_map_path": os.path.abspath(tx_map_path),
            "rx_map_path": os.path.abspath(rx_map_path),
        })

    fieldnames = [
        "tx_id",
        "tx_x", "tx_y", "tx_z",
        "ris_x", "ris_y", "ris_z",
        "ris_roll", "ris_pitch", "ris_yaw",
        "rx_x", "rx_y", "rx_z",
        "power_map_path",
        "tx_map_path",
        "rx_map_path",
    ]
    with open(CSV_OUTPUT_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nDone. Processed {len(csv_rows)} entries.")
    print(f"CSV written to: {CSV_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
