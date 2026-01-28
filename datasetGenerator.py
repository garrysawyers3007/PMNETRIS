import os
import json
import numpy as np
import sionna as sn
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, RIS, Camera
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

# =================================================================================
# 1. CONFIGURATION
# =================================================================================
ROOT = "Data/"
TX_POS_FILE = f"{ROOT}tx_positions.npy"
SCENE_FILE = f"{ROOT}USC_3D/USC.xml"
JSON_INPUT_FILE = "dataset_32x32.json"

# Output directories
OUTPUT_SCENE_DIR = f"{ROOT}scenes"
OUTPUT_IMG_DIR = f"{ROOT}gain_images" # New directory for gain images
OUTPUT_RIS_SCENE_DIR = f"{ROOT}scenes_ris"

# Simulation parameters
CM_CELL_SIZE = (5.0, 5.0)
MAX_DEPTH = 1000      # High quality
NUM_SAMPLES = int(2e6) # High quality

# RIS parameters
RIS_NUM_ROWS = 32
RIS_NUM_COLS = 32
RIS_NUM_MODES = 1
RIS_INITIAL_ORIENTATION_EULER = np.array([-np.pi / 2, 0.0, 0.0])

# =================================================================================
# 2. HELPER FUNCTIONS
# =================================================================================
def get_scene(xml_file):
    scene = load_scene(xml_file)

    scene.tx_array = PlanarArray(
    num_rows = 4,
    num_cols = 4,
    vertical_spacing = 0.5,
    horizontal_spacing = 0.5,
    pattern = "iso",
    polarization="V"
    )
    scene.rx_array = PlanarArray(
    num_rows = 1,
    num_cols = 1,
    vertical_spacing = 0.5,
    horizontal_spacing = 0.5,
    pattern = "iso",
    polarization="V"
    )
    accepted_mat = ["itu_concrete", "itu_very_dry_ground"]
    for obj_name in scene.objects:
        obj = scene.get(obj_name)
        if scene.get(obj_name).radio_material.name not in accepted_mat:
            obj.radio_material = "itu_concrete"

    return scene

def save_gain_plot(gain_map_db, filename):
    """Saves the gain map visualization using Matplotlib."""
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(gain_map_db, origin='lower', vmin=0) # origin='lower' matches your request
    plt.colorbar(label='Gain [dB]')
    plt.xlabel('Cell index (X-axis)')
    plt.ylabel('Cell index (Y-axis)')
    plt.title("RIS Coverage Gain")
    plt.savefig(filename)
    plt.close(fig)

# =================================================================================
# 3. MAIN EXECUTION
# =================================================================================
def main():
    if not os.path.exists(JSON_INPUT_FILE): print(f"Error: {JSON_INPUT_FILE} not found."); return
    if not os.path.exists(TX_POS_FILE): print(f"Error: {TX_POS_FILE} not found."); return

    with open(JSON_INPUT_FILE, 'r') as f: dataset = json.load(f)
    TX = np.load(TX_POS_FILE)
    
    os.makedirs(OUTPUT_SCENE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_RIS_SCENE_DIR, exist_ok=True)
    
    print(f"Generating RF maps for {len(dataset)} transmitters found in {JSON_INPUT_FILE}...")

    for entry in tqdm(dataset, desc="Processing Transmitters"):
        tx_id_str = entry.get('tx_id')
        if tx_id_str is None: continue
        try: tx_idx = int(tx_id_str)
        except ValueError: continue
        if tx_idx < 0 or tx_idx >= len(TX): continue

        tx_pos_3d = TX[tx_idx]
        tx_pos_3d[2] += 2.0  # Elevate TX by 2 meters

        # 1. Setup Scene & Baseline (No RIS)
        scene = get_scene(SCENE_FILE)
        tx_name = f"tx_{tx_id_str}"
        tx = Transmitter(name=tx_name, position=tx_pos_3d)
        scene.add(tx)
        
        # Compute No-RIS Map
        cm_no_ris = scene.coverage_map(
            cm_cell_size=CM_CELL_SIZE, diffraction=True, scattering=True, edge_diffraction=True, 
            max_depth=MAX_DEPTH, num_samples=NUM_SAMPLES
        )
        path_gain_no_ris = cm_no_ris.path_gain[0].numpy()

        # Render Baseline Scene Image (Optional, kept from your snippet)
        cam_pos = [tx_pos_3d[0], tx_pos_3d[1] - 0.01, 1500.0]
        bird_cam_name = f"birds_view_{tx_id_str}"
        bird_cam = Camera(bird_cam_name, position=cam_pos, look_at=tx_pos_3d)
        scene.add(bird_cam)
        scene.render_to_file(camera=bird_cam_name, coverage_map=cm_no_ris, 
                             filename=os.path.join(OUTPUT_SCENE_DIR, f"scene_{tx_id_str}.png"), num_samples=512)
        
        tx_ris_subdir = os.path.join(OUTPUT_RIS_SCENE_DIR, f"tx_{tx_id_str}")
        tx_gain_img_subdir = os.path.join(OUTPUT_IMG_DIR, f"tx_{tx_id_str}")
        os.makedirs(tx_ris_subdir, exist_ok=True)
        os.makedirs(tx_gain_img_subdir, exist_ok=True)

        # 2. Iterate RIS Records
        for i, record in enumerate(entry['records']):
            ris_pos = record['RIS']
            rx_pos = record['RX']
            ris_name = f"ris_{tx_id_str}_{i}"
            rx_name = f"rx_steer_{tx_id_str}_{i}"
            
            rx_steer = Receiver(name=rx_name, position=rx_pos)
            scene.add(rx_steer)
            ris = RIS(name=ris_name, position=ris_pos, orientation=RIS_INITIAL_ORIENTATION_EULER,
                      num_rows=RIS_NUM_ROWS, num_cols=RIS_NUM_COLS, num_modes=RIS_NUM_MODES)
            scene.add(ris)
            
            ris.look_at((tx.position + rx_steer.position) / 2)
            ris.phase_gradient_reflector(tx.position, rx_steer.position)

            # Compute RIS Map
            cm_ris = scene.coverage_map(
                cm_cell_size=CM_CELL_SIZE, ris=True, diffraction=True, scattering=True, edge_diffraction=True, 
                max_depth=MAX_DEPTH, num_samples=NUM_SAMPLES
            )
            path_gain_ris = cm_ris.path_gain[0].numpy()

            # scene.render_to_file(camera=bird_cam_name, coverage_map=cm_ris, 
            #                      filename=os.path.join(tx_ris_subdir, f"rec_{i}.png"), num_samples=512)
            cm_ris.show(show_ris=True, show_rx=True).savefig(os.path.join(tx_ris_subdir, f"rec_{i}.png"))
            
            # 3. Compute Gain Map
            gain_map_db = 10 * np.log10(path_gain_ris / (path_gain_no_ris + 1e-20))
            
            # 4. Save Outputs
            base_fname = f"rec_{i}"
            save_gain_plot(gain_map_db, os.path.join(tx_gain_img_subdir, f"{base_fname}_gain.png"))
            
            scene.remove(ris_name); scene.remove(rx_name)

    print("Generation complete.")

if __name__ == "__main__":

    main()