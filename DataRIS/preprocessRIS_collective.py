import numpy as np
import sionna as sn
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, RIS, Scene
import cv2
import os
import json
import sys

# Define constants (adjust as needed)
ROOT = "DataRIS/"
OUTPUT_ROOT = "datasetGeoRIS_collective/"
TX_POS_FILE = f"{ROOT}tx_positions.npy"
SCENE_FILE = f"{ROOT}USC_3D/USC.xml"
CITY_MAP_FILE = f"{ROOT}USC_city_map.png"
JSON_INPUT_FILE = "dataset_new_32x32.json"
JSON_OUTPUT_FILE = f"{OUTPUT_ROOT}metadata.json"

CM_DIM = 900  # Coverage map dimension (pixels)
NEW_CM_DIM = 10
CM_CELL_SIZE = (5.0, 5.0)  # Cell size in meters
TX_WIDTH = 8   # TX marker width in pixels
FLOOR = -200   # Min power value (dB)

# RIS parameters
RIS_NUM_ROWS = 32
RIS_NUM_COLS = 32
RIS_NUM_MODES = 1
RIS_INITIAL_ORIENTATION_EULER = np.array([-np.pi/2, 0.0, 0.0])  # [roll, pitch, yaw]


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
        if scene.get(obj_name).radio_material.name not in accepted_mat:
            obj.radio_material = "itu_concrete"
    return scene


def augment_image(image):
    """Generates original and 3 flipped versions of an image."""
    flipped_vertical = cv2.flip(image, 0)
    flipped_horizontal = cv2.flip(image, 1)
    flipped_both = cv2.flip(image, -1)
    return image, flipped_vertical, flipped_horizontal, flipped_both


def augment_euler_orientation_ordered(world_rpy_array_rad):
    R, P, Y = world_rpy_array_rad
    orientations = [
        np.array([R,  P, Y]),   # image_0: original
        np.array([R, -P, Y]),   # image_1: vertical flip
        np.array([-R,  P, Y]),  # image_2: horizontal flip
        np.array([-R, -P, Y]),  # image_3: both flips
    ]
    return [arr.tolist() for arr in orientations]


def augment_position_ordered(world_xyz_array):
    x, y, z = world_xyz_array
    positions = [
        np.array([ x,  y, z]),  # image_0: original
        np.array([ x, -y, z]),  # image_1: vertical flip
        np.array([-x,  y, z]),  # image_2: horizontal flip
        np.array([-x, -y, z]),  # image_3: both flips
    ]
    return [arr.tolist() for arr in positions]


def get_sweep_positions_3d(cm_center_3d, n_dim=NEW_CM_DIM, cell_size=CM_CELL_SIZE[0]):
    """
    Returns world-space 3D positions for all diagonal and cross-diagonal pixels
    of the n_dim x n_dim coverage map centered at cm_center_3d.

    Uses half-integer offsets so each position corresponds to a pixel center.
    For n_dim=10 (even), diagonal and anti-diagonal never coincide -> 20 unique positions.
    The equivalent step size along the diagonal direction is cell_size * sqrt(2) meters.
    """
    cx, cy, cz = float(cm_center_3d[0]), float(cm_center_3d[1]), float(cm_center_3d[2])
    # Half-integer offsets: [-4.5, -3.5, ..., 4.5] for n_dim=10
    offsets = np.arange(n_dim) - (n_dim - 1) / 2.0

    seen = set()
    positions = []
    for k in offsets:
        dx = float(k) * cell_size
        # Main diagonal pixel (col=k, row=k)
        p_diag = (cx + dx, cy + dx, cz)
        if p_diag not in seen:
            seen.add(p_diag)
            positions.append(list(p_diag))
        # Anti-diagonal pixel (col=k, row=-k)
        p_anti = (cx + dx, cy - dx, cz)
        if p_anti not in seen:
            seen.add(p_anti)
            positions.append(list(p_anti))
    return positions


def generate_raw_power_map(scene_to_use, coverage_params):
    """
    Calls scene.coverage_map and returns a raw float32 dB array.
    Applies: 10*log10, replace -inf with -255, clip values > 0 to 0.
    Does NOT apply: flip, floor, +255 shift, or uint8 cast.
    Those are applied once to the collective max via postprocess_power_map().
    Returns array of shape (NEW_CM_DIM, NEW_CM_DIM), float32.
    """
    cm = scene_to_use.coverage_map(**coverage_params)
    if cm.path_gain.shape[0] == 0:
        return np.full((NEW_CM_DIM, NEW_CM_DIM), -255.0, dtype=np.float32)
    tx_cm = (10.0 * np.log10(cm.path_gain[0].numpy())).astype(np.float32)
    tx_cm[tx_cm == (-np.inf)] = -255.0
    tx_cm[tx_cm > 0] = 0.0
    return tx_cm


def postprocess_power_map(raw_float_cm):
    """
    Applies final post-processing once to the collective raw float dB map:
    vertical flip, floor at FLOOR, shift by +255, clip to uint8.
    """
    cm = cv2.flip(raw_float_cm, 0)
    cm[cm < FLOOR] = FLOOR
    cm += 255
    return np.clip(cm.astype(np.uint8), 0, 255)


def process_and_save_maps(scene_to_use, tx_id_str, tx_pos_xy_for_txmap, city_map_img,
                          output_folder_path, file_prefix_str, metadata_list_ref,
                          is_ris_scenario=False, ris_info_dict=None, rx_center=None,
                          precomputed_power_map=None):
    """
    Generates (or uses a precomputed), processes, flips, and saves maps, updating metadata.

    For RIS scenarios pass precomputed_power_map (already uint8, shape NEW_CM_DIM x NEW_CM_DIM)
    to skip internal coverage map generation.  For noRIS scenarios leave it None and the
    original cm generation + postprocessing pipeline runs as usual.
    """
    print(f"  Generating coverage map for {file_prefix_str}...")
    if not scene_to_use.rx_array:
        scene_to_use.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

    if precomputed_power_map is not None:
        # Collective RIS map already computed and postprocessed externally
        tx_cm_processed = precomputed_power_map
    else:
        # Original single-steer coverage map generation (used for noRIS case)
        coverage_map_params = {
            "cm_cell_size": CM_CELL_SIZE,
            "diffraction": True, "scattering": True, "edge_diffraction": True,
        }
        if is_ris_scenario:
            coverage_map_params["ris"] = True
            coverage_map_params["max_depth"] = 1000
            coverage_map_params["num_samples"] = int(2*(10**6))
            coverage_map_params["cm_center"] = ris_info_dict["rx_pos_for_steering"]
            coverage_map_params["cm_size"] = [50, 50]
            coverage_map_params["cm_orientation"] = [0, 0, 0]
        else:
            coverage_map_params["max_depth"] = 1000
            coverage_map_params["num_samples"] = int(2*(10**6))
            if rx_center is not None:
                coverage_map_params["cm_center"] = rx_center
                coverage_map_params["cm_size"] = [50, 50]
                coverage_map_params["cm_orientation"] = [0, 0, 0]

        cm = scene_to_use.coverage_map(**coverage_map_params)
        if cm.path_gain.shape[0] == 0:
            print(f"  Warning: No path gain data for {file_prefix_str}. Saving blank power map.")
            tx_cm_processed = np.zeros((NEW_CM_DIM, NEW_CM_DIM), dtype=np.uint8)
        else:
            tx_cm = 10.*np.log10(cm.path_gain[0].numpy())
            tx_cm[tx_cm == (-np.inf)] = -255
            tx_cm[tx_cm > 0] = 0
            tx_cm = cv2.flip(tx_cm, 0)
            tx_cm[tx_cm < FLOOR] = FLOOR
            tx_cm += 255
            tx_cm_processed = np.clip(tx_cm.astype(np.uint8), 0, 255)

    # --- Build marker maps ---
    tx_pos_2d_map = (tx_pos_xy_for_txmap[:2] + (CM_DIM // 2) + 50).astype(np.int16)
    tx_map_img = np.zeros((CM_DIM, CM_DIM), dtype=np.uint8)
    shift = TX_WIDTH // 2
    map_y_center_tx = CM_DIM - tx_pos_2d_map[1]
    y_start = np.clip(map_y_center_tx - shift, 0, CM_DIM); y_end = np.clip(map_y_center_tx + shift + 1, 0, CM_DIM)
    x_start = np.clip(tx_pos_2d_map[0] - shift, 0, CM_DIM); x_end = np.clip(tx_pos_2d_map[0] + shift + 1, 0, CM_DIM)
    tx_map_img[y_start:y_end, x_start:x_end] = 255

    rx_map_img = np.zeros((CM_DIM, CM_DIM), dtype=np.uint8)
    if is_ris_scenario and ris_info_dict:
        rx_pos_3d = np.array(ris_info_dict["rx_pos_for_steering"])
        rx_pos_2d_map = (rx_pos_3d[:2] + (CM_DIM // 2) + 50).astype(np.int16)
        map_y_center_rx = CM_DIM - rx_pos_2d_map[1]
        y_start_rx = np.clip(map_y_center_rx - shift, 0, CM_DIM); y_end_rx = np.clip(map_y_center_rx + shift + 1, 0, CM_DIM)
        x_start_rx = np.clip(rx_pos_2d_map[0] - shift, 0, CM_DIM); x_end_rx = np.clip(rx_pos_2d_map[0] + shift + 1, 0, CM_DIM)
        rx_map_img[y_start_rx:y_end_rx, x_start_rx:x_end_rx] = 255
    elif not is_ris_scenario and rx_center is not None:
        rx_pos_3d = np.array(rx_center)
        rx_pos_2d_map = (rx_pos_3d[:2] + (CM_DIM // 2) + 50).astype(np.int16)
        map_y_center_rx = CM_DIM - rx_pos_2d_map[1]
        y_start_rx = np.clip(map_y_center_rx - shift, 0, CM_DIM); y_end_rx = np.clip(map_y_center_rx + shift + 1, 0, CM_DIM)
        x_start_rx = np.clip(rx_pos_2d_map[0] - shift, 0, CM_DIM); x_end_rx = np.clip(rx_pos_2d_map[0] + shift + 1, 0, CM_DIM)
        rx_map_img[y_start_rx:y_end_rx, x_start_rx:x_end_rx] = 255

    ris_map_img = np.zeros((CM_DIM, CM_DIM), dtype=np.uint8)
    if is_ris_scenario and ris_info_dict:
        ris_pos_3d = np.array(ris_info_dict["ris_pos"])
        ris_pos_2d_map = (ris_pos_3d[:2] + (CM_DIM // 2) + 50).astype(np.int16)
        map_y_center_ris = CM_DIM - ris_pos_2d_map[1]
        y_start_ris = np.clip(map_y_center_ris - shift, 0, CM_DIM); y_end_ris = np.clip(map_y_center_ris + shift + 1, 0, CM_DIM)
        x_start_ris = np.clip(ris_pos_2d_map[0] - shift, 0, CM_DIM); x_end_ris = np.clip(ris_pos_2d_map[0] + shift + 1, 0, CM_DIM)
        ris_map_img[y_start_ris:y_end_ris, x_start_ris:x_end_ris] = 255

    maps_to_save_dict = {
        "tx_map": tx_map_img.copy(), "rx_map": rx_map_img.copy(),
        "ris_map": ris_map_img.copy(), "power_map": tx_cm_processed.copy(),
        "city_map": city_map_img.copy()
    }
    paths_data_dict = {}
    for map_type_str, map_image_data in maps_to_save_dict.items():
        augmented_maps_tuple = augment_image(map_image_data)
        paths_data_dict[map_type_str] = []
        for i, augmented_map_image in enumerate(augmented_maps_tuple):
            file_path_str = os.path.join(output_folder_path, f"{file_prefix_str}_{map_type_str}_{i}.png")
            cv2.imwrite(file_path_str, augmented_map_image)
            paths_data_dict[map_type_str].append(os.path.abspath(file_path_str))

    record_data_dict = {"tx_id": tx_id_str, "type": "RIS" if is_ris_scenario else "noRIS", "paths": paths_data_dict}
    if is_ris_scenario and ris_info_dict:
        true_ris_pos = ris_info_dict["ris_pos"]
        record_data_dict["ris_true_world_pos"] = true_ris_pos
        record_data_dict["ris_positions_xyz_augmented_for_view"] = augment_position_ordered(np.array(true_ris_pos))

        true_rx_pos = ris_info_dict["rx_pos_for_steering"]
        record_data_dict["rx_true_world_pos_for_steering"] = true_rx_pos
        record_data_dict["rx_positions_xyz_augmented_for_view"] = augment_position_ordered(np.array(true_rx_pos))

        record_data_dict["ris_true_world_orientation_rpy"] = ris_info_dict["ris_actual_orientation_world_rpy"]
        record_data_dict["orientations_rpy_augmented_for_view"] = augment_euler_orientation_ordered(
            np.array(ris_info_dict["ris_actual_orientation_world_rpy"]))

        record_data_dict["record_index"] = ris_info_dict["index"]
    else:
        record_data_dict.update({
            "record_index": None, "ris_true_world_pos": None, "ris_positions_xyz_augmented_for_view": None,
            "rx_true_world_pos_for_steering": None, "rx_positions_xyz_augmented_for_view": None,
            "ris_true_world_orientation_rpy": None, "orientations_rpy_augmented_for_view": None
        })
    metadata_list_ref.append(record_data_dict)
    print(f"  Saved maps for {file_prefix_str}.")


def main(start=None, end=None):
    json_output_file = f"{OUTPUT_ROOT}metadata_{start}_{end}.json" if (start is not None and end is not None) else f"{OUTPUT_ROOT}metadata.json"
    print("Starting collective dataset generation...")
    try:
        with open(JSON_INPUT_FILE, 'r') as f: data = json.load(f)
        tx_positions_all = np.load(TX_POS_FILE)
        city_map_original = cv2.imread(CITY_MAP_FILE)[:, :, 0]
        if city_map_original is None: raise FileNotFoundError(f"{CITY_MAP_FILE} missing/invalid.")
        city_map_resized = cv2.resize(city_map_original, (CM_DIM, CM_DIM))
        print("Input files loaded.")
    except FileNotFoundError as e: print(f"Error loading input files: {e}"); return
    except Exception as e: print(f"An unexpected error during input loading: {e}"); return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    metadata_collector = []

    for tx_entry_item in data:
        # Filter by TX ID range if specified
        try:
            tx_id_int = int(tx_entry_item["tx_id"])
        except ValueError:
            tx_id_int = -1
        if start is not None and tx_id_int < start:
            continue
        if end is not None and tx_id_int >= end:
            continue
        current_tx_id_str = tx_entry_item["tx_id"]
        print(f"Processing TX_ID: {current_tx_id_str}")
        try: current_tx_id_int = int(current_tx_id_str)
        except ValueError: print(f"  Warning: Invalid TX_ID '{current_tx_id_str}'. Skipping."); continue
        if not (0 <= current_tx_id_int < len(tx_positions_all)):
            print(f"  Warning: TX_ID {current_tx_id_int} out of bounds. Skipping."); continue

        current_tx_pos = tx_positions_all[current_tx_id_int]
        print(f"  TX Position: {current_tx_pos}")
        current_tx_pos_3d_scene = np.array([current_tx_pos[0], current_tx_pos[1], current_tx_pos[2]])
        tx_specific_output_folder = os.path.join(OUTPUT_ROOT, current_tx_id_str)
        os.makedirs(tx_specific_output_folder, exist_ok=True)

        # --- No RIS Case (unchanged) ---
        print("  Processing 'no RIS' cases...")
        unique_rx_positions = {}
        for i, record_item in enumerate(tx_entry_item["records"]):
            rx_pos = tuple(record_item["RX"])
            if rx_pos not in unique_rx_positions:
                unique_rx_positions[rx_pos] = i

        for rx_idx, (rx_pos_tuple, record_idx) in enumerate(unique_rx_positions.items()):
            rx_pos_list = list(rx_pos_tuple)
            scene_no_ris = get_scene(SCENE_FILE)
            tx_obj_name_no_ris = f"tx{current_tx_id_str}_no_ris_rx{rx_idx}"
            transmitter_no_ris = Transmitter(tx_obj_name_no_ris, current_tx_pos_3d_scene + [0.0, 0.0, 2.0], [0.0, 0.0, 0.0])
            scene_no_ris.add(transmitter_no_ris)
            process_and_save_maps(scene_no_ris, current_tx_id_str, current_tx_pos, city_map_resized,
                                  tx_specific_output_folder, f"{current_tx_id_str}_noRIS_rx{rx_idx}", metadata_collector,
                                  is_ris_scenario=False, rx_center=rx_pos_list)

        # --- RIS Cases (collective sweep) ---
        print(f"  Processing {len(tx_entry_item['records'])} RIS cases for TX_ID {current_tx_id_str}...")
        for i, record_item in enumerate(tx_entry_item["records"]):
            ris_position_3d = np.array(record_item["RIS"])
            rx_pos_for_steering_3d = np.array(record_item["RX"])
            current_ris_name = f"ris_{current_tx_id_str}_{i}"
            temp_rx_name_for_steering = f"rx_steer_{current_tx_id_str}_{i}"
            tx_obj_name_ris = f"tx{current_tx_id_str}_ris_case_{i}"

            print(f"    Record {i}: RIS at {ris_position_3d}, RX_center at {rx_pos_for_steering_3d}")
            scene_with_ris = get_scene(SCENE_FILE)
            transmitter_ris_case = Transmitter(tx_obj_name_ris, current_tx_pos_3d_scene + [0.0, 0.0, 2.0], [0.0, 0.0, 0.0])
            scene_with_ris.add(transmitter_ris_case)
            temp_rx_for_steering = Receiver(temp_rx_name_for_steering, rx_pos_for_steering_3d)
            scene_with_ris.add(temp_rx_for_steering)

            ris_object = RIS(name=current_ris_name, position=ris_position_3d,
                             orientation=RIS_INITIAL_ORIENTATION_EULER,
                             num_rows=RIS_NUM_ROWS, num_cols=RIS_NUM_COLS,
                             num_modes=RIS_NUM_MODES)
            ris_object.look_at((transmitter_ris_case.position + temp_rx_for_steering.position) / 2.0)
            scene_with_ris.add(ris_object)

            # Capture physical orientation once (look_at sets it; phase_gradient_reflector only
            # updates the phase codebook, not the surface orientation)
            actual_ris_orientation_euler = ris_object.orientation.numpy()

            # Fixed coverage map params — cm_center stays at the original RX position for all sweeps
            cm_params = {
                "cm_cell_size": CM_CELL_SIZE,
                "diffraction": True, "scattering": True, "edge_diffraction": True,
                "ris": True,
                "max_depth": 1000,
                "num_samples": int(2*(10**6)),
                "cm_center": rx_pos_for_steering_3d.tolist(),
                "cm_size": [50, 50],
                "cm_orientation": [0, 0, 0],
            }

            # Sweep phase_gradient_reflector over diagonal + cross-diagonal pixel centers
            sweep_positions = get_sweep_positions_3d(rx_pos_for_steering_3d)
            print(f"    Sweeping {len(sweep_positions)} steering targets (diagonal + cross-diagonal)...")
            collective_raw = None
            for j, sweep_pos in enumerate(sweep_positions):
                print(f"      Sweep {j+1}/{len(sweep_positions)}: steering to {np.round(sweep_pos, 1)}")
                ris_object.phase_gradient_reflector(transmitter_ris_case.position, np.array(sweep_pos))
                raw_map = generate_raw_power_map(scene_with_ris, cm_params)
                collective_raw = raw_map if collective_raw is None else np.maximum(collective_raw, raw_map)

            # Apply flip / floor / +255 / uint8 ONCE to the collective max map
            collective_power_map = postprocess_power_map(collective_raw)

            ris_info_for_metadata = {
                "index": i, "ris_pos": ris_position_3d.tolist(),
                "rx_pos_for_steering": rx_pos_for_steering_3d.tolist(),
                "ris_actual_orientation_world_rpy": actual_ris_orientation_euler.tolist()
            }
            process_and_save_maps(scene_with_ris, current_tx_id_str, current_tx_pos, city_map_resized,
                                  tx_specific_output_folder, f"{current_tx_id_str}_RIS_{i}", metadata_collector,
                                  is_ris_scenario=True, ris_info_dict=ris_info_for_metadata,
                                  precomputed_power_map=collective_power_map)

    print(f"Saving metadata to {json_output_file}...")
    try:
        with open(json_output_file, 'w') as f: json.dump(metadata_collector, f, indent=2)
    except Exception as e: print(f"Error saving metadata JSON: {e}")
    print("Collective dataset generation complete.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocessRIS_collective.py <start_tx_id> <end_tx_id>")
        print("  start_tx_id : inclusive start of TX ID range")
        print("  end_tx_id   : exclusive end of TX ID range")
        sys.exit(1)
    arg_start = int(sys.argv[1])
    arg_end   = int(sys.argv[2])
    if arg_start < 0:
        print("start_tx_id must be >= 0"); sys.exit(1)
    if arg_end <= arg_start:
        print("end_tx_id must be greater than start_tx_id"); sys.exit(1)
    print(f"Processing TX IDs [{arg_start}, {arg_end})...")

    if not os.path.exists(ROOT): os.makedirs(ROOT)
    dummy_3d_path = os.path.join(ROOT, "USC_3D")
    if not os.path.exists(dummy_3d_path): os.makedirs(dummy_3d_path)
    if not os.path.exists(JSON_INPUT_FILE):
        print(f"Creating dummy '{JSON_INPUT_FILE}'...")
        with open(JSON_INPUT_FILE, 'w') as f: json.dump([{"tx_id": "0", "records": [{"RIS": [-60, 300, 14], "RX": [45, 250, 14]}]}], f)
    if not os.path.exists(TX_POS_FILE):
        print(f"Creating dummy '{TX_POS_FILE}'...")
        np.save(TX_POS_FILE, np.array([[100, 100, 0], [-50, 150, 0]]))
    if not os.path.exists(CITY_MAP_FILE):
        print(f"Creating dummy '{CITY_MAP_FILE}'...")
        cv2.imwrite(CITY_MAP_FILE, np.full((CM_DIM, CM_DIM), 128, dtype=np.uint8))
    scene_xml_path = os.path.join(dummy_3d_path, "USC.xml")
    if not os.path.exists(scene_xml_path):
        print(f"Creating dummy '{scene_xml_path}'...")
        with open(scene_xml_path, 'w') as f:
            f.write("""<?xml version="1.0"?><scene version="0.6.0"><integrator type="path"/><shape type="rectangle"><bsdf type="diffuse"><string name="radio_material" value="itu_very_dry_ground"/></bsdf></shape></scene>""")

    main(start=arg_start, end=arg_end)
