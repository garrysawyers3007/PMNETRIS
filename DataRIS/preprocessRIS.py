import numpy as np
import sionna as sn
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, RIS, Scene
import cv2
import os
import json

# Define constants (adjust as needed)
ROOT = "DataRIS/"
OUTPUT_ROOT = "datasetGeoRIS/"
TX_POS_FILE = f"{ROOT}tx_positions.npy"
SCENE_FILE = f"{ROOT}USC_3D/USC.xml"
CITY_MAP_FILE = f"{ROOT}USC_city_map.png"
JSON_INPUT_FILE = "dataset_new_32x32.json"
JSON_OUTPUT_FILE = f"{OUTPUT_ROOT}metadata.json"

CM_DIM = 900  # Coverage map dimension (pixels)
NEW_CM_DIM = 10
CM_CELL_SIZE = (5.0, 5.0) # Cell size in meters
TX_WIDTH = 8  # TX marker width in pixels
FLOOR = -200 # Min power value (dB)

# RIS parameters from user snippet
RIS_NUM_ROWS = 32
RIS_NUM_COLS = 32
RIS_NUM_MODES = 1
# Initial orientation for RIS object creation, will be overridden by look_at
RIS_INITIAL_ORIENTATION_EULER = np.array([-np.pi/2, 0.0, 0.0]) # [roll, pitch, yaw]

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
def augment_image(image):
    """Generates original and 3 flipped versions of an image."""
    flipped_vertical = cv2.flip(image, 0)
    flipped_horizontal = cv2.flip(image, 1)
    flipped_both = cv2.flip(image, -1)
    # Order: [original_processed, vertical_flip, horizontal_flip, both_flip]
    return image, flipped_vertical, flipped_horizontal, flipped_both

def augment_euler_orientation_ordered(world_rpy_array_rad):
    """
    Transforms true world Euler angles [R,P,Y] (roll, pitch, yaw in radians)
    to correspond with the four augmented 2D image views for metadata.
    These transformations correspond to rotating the viewing frame.
    """
    R, P, Y = world_rpy_array_rad # True world orientation

    # For image_0 (original processed map). View: X_world, -Y_world_view (approx Rx(180 deg) world rot)
    orientation_idx0 = np.array([R, P, Y])

    # For image_1 (vertically flipped map). View: X_world, Y_world_view (canonical world view)
    orientation_idx1 = np.array([R, -P, Y])

    # For image_2 (horizontally flipped map). View: -X_world, -Y_world_view (approx Rz(180 deg) world rot)
    orientation_idx2 = np.array([-R, P, Y])

    # For image_3 (both flipped map). View: -X_world, Y_world_view (approx Ry(180 deg) world rot)
    orientation_idx3 = np.array([-R, -P, Y])
    
    orientations = [
        orientation_idx0,
        orientation_idx1,
        orientation_idx2,
        orientation_idx3
    ]
    # Normalize angles to [-pi, pi] - important for consistency if needed later
    # For now, direct transformation is stored.
    # e.g., angles = (angles + np.pi) % (2 * np.pi) - np.pi
    return [arr.tolist() for arr in orientations]

def augment_position_ordered(world_xyz_array):
    """
    Transforms a true world 3D coordinate [x, y, z] to correspond with
    the four augmented 2D image views for metadata.

    These transformations correspond to rotating the viewing frame and match
    the logic used for augmenting Euler orientations.
    """
    x, y, z = world_xyz_array # True world position

    # The order of transformations matches the output of augment_image:
    # [original_processed, vertical_flip, horizontal_flip, both_flip]

    # For image_0 (original processed map). View corresponds to a world
    # rotated 180 degrees around its X-axis.
    # Transformation: (x, y, z) -> (x, -y, -z)
    position_idx0 = np.array([x, y, z])

    # For image_1 (vertically flipped map). This is the canonical world view.
    # Transformation: (x, y, z) -> (x, y, z)
    position_idx1 = np.array([x, -y, z])

    # For image_2 (horizontally and vertically flipped map). View corresponds to a
    # world rotated 180 degrees around its Z-axis.
    # Transformation: (x, y, z) -> (-x, -y, z)
    position_idx2 = np.array([-x, y, z])

    # For image_3 (horizontally flipped map). View corresponds to a world
    # rotated 180 degrees around its Y-axis.
    # Transformation: (x, y, z) -> (-x, y, -z)
    position_idx3 = np.array([-x, -y, z])

    positions = [
        position_idx0,
        position_idx1,
        position_idx2,
        position_idx3
    ]

    return [arr.tolist() for arr in positions]


def process_and_save_maps(scene_to_use, tx_id_str, tx_pos_xy_for_txmap, city_map_img,
                          output_folder_path, file_prefix_str, metadata_list_ref,
                          is_ris_scenario=False, ris_info_dict=None, rx_center=None):
    """Generates, processes, flips, and saves maps, updating metadata."""
    print(f"  Generating coverage map for {file_prefix_str}...")
    if not scene_to_use.rx_array:
        scene_to_use.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

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
        coverage_map_params["max_depth"] = 1000 # Default for non-RIS
        coverage_map_params["num_samples"] = int(2*(10**6)) # Default for non-RIS
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
        
        tx_cm[tx_cm==(-np.inf)] = -255
        tx_cm[tx_cm > 0] = 0
        # tx_cm = cv2.resize(tx_cm, city_map_img.shape)

        # tx_cm = cv2.flip(tx_cm, 0)
        tx_cm[tx_cm < FLOOR] = FLOOR
        tx_cm += 255

        # tx_cm[city_map_img < 55] = 0 # for building
        
        # tx_cm_raw = 10. * np.log10(cm.path_gain[0].numpy())
        # tx_cm_raw[tx_cm_raw==(-np.inf)] = -255
        # tx_cm_raw[tx_cm_raw > 0] = 0
        # tx_cm_resized = cv2.resize(tx_cm_raw, city_map_img.shape)
        # tx_cm_flipped_coords = cv2.flip(tx_cm_resized, 0) # This is image_0 base
        # tx_cm_floored = tx_cm_flipped_coords.copy()
        # tx_cm_floored[tx_cm_floored < FLOOR] = FLOOR
        # tx_cm_normalized = tx_cm_floored + 255
        # mask = city_map_img < 55
        # tx_cm_normalized[mask] = 0
        tx_cm_processed = np.clip(tx_cm.astype(np.uint8), 0, 255)

    tx_pos_2d_map = (tx_pos_xy_for_txmap[:2] + (CM_DIM // 2) + (50)).astype(np.int16)
    tx_map_img = np.zeros((CM_DIM, CM_DIM), dtype=np.uint8)
    shift = TX_WIDTH // 2
    map_y_center_tx = CM_DIM - tx_pos_2d_map[1]
    y_start = np.clip(map_y_center_tx - shift, 0, CM_DIM); y_end = np.clip(map_y_center_tx + shift + 1, 0, CM_DIM)
    x_start = np.clip(tx_pos_2d_map[0] - shift, 0, CM_DIM); x_end = np.clip(tx_pos_2d_map[0] + shift + 1, 0, CM_DIM)
    tx_map_img[y_start:y_end, x_start:x_end] = 255

    # Create RX map (for both RIS and non-RIS scenarios)
    rx_map_img = np.zeros((CM_DIM, CM_DIM), dtype=np.uint8)
    if is_ris_scenario and ris_info_dict:
        rx_pos_3d = np.array(ris_info_dict["rx_pos_for_steering"])
        rx_pos_2d_map = (rx_pos_3d[:2] + (CM_DIM // 2) + (50)).astype(np.int16)
        map_y_center_rx = CM_DIM - rx_pos_2d_map[1]
        y_start_rx = np.clip(map_y_center_rx - shift, 0, CM_DIM); y_end_rx = np.clip(map_y_center_rx + shift + 1, 0, CM_DIM)
        x_start_rx = np.clip(rx_pos_2d_map[0] - shift, 0, CM_DIM); x_end_rx = np.clip(rx_pos_2d_map[0] + shift + 1, 0, CM_DIM)
        rx_map_img[y_start_rx:y_end_rx, x_start_rx:x_end_rx] = 255
    elif not is_ris_scenario and rx_center is not None:
        rx_pos_3d = np.array(rx_center)
        rx_pos_2d_map = (rx_pos_3d[:2] + (CM_DIM // 2) + (50)).astype(np.int16)
        map_y_center_rx = CM_DIM - rx_pos_2d_map[1]
        y_start_rx = np.clip(map_y_center_rx - shift, 0, CM_DIM); y_end_rx = np.clip(map_y_center_rx + shift + 1, 0, CM_DIM)
        x_start_rx = np.clip(rx_pos_2d_map[0] - shift, 0, CM_DIM); x_end_rx = np.clip(rx_pos_2d_map[0] + shift + 1, 0, CM_DIM)
        rx_map_img[y_start_rx:y_end_rx, x_start_rx:x_end_rx] = 255

    maps_to_save_dict = {"tx_map": tx_map_img.copy(), "rx_map": rx_map_img.copy(), "power_map": tx_cm_processed.copy(), "city_map": city_map_img.copy()}
    paths_data_dict = {}

    for map_type_str, map_image_data in maps_to_save_dict.items():
        augmented_maps_tuple = augment_image(map_image_data) # Returns 4 images
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
        
        # RX Position (NEW)
        true_rx_pos = ris_info_dict["rx_pos_for_steering"]
        record_data_dict["rx_true_world_pos_for_steering"] = true_rx_pos
        record_data_dict["rx_positions_xyz_augmented_for_view"] = augment_position_ordered(np.array(true_rx_pos))

        # RIS Orientation
        record_data_dict["ris_true_world_orientation_rpy"] = ris_info_dict["ris_actual_orientation_world_rpy"]
        record_data_dict["orientations_rpy_augmented_for_view"] = augment_euler_orientation_ordered(np.array(ris_info_dict["ris_actual_orientation_world_rpy"]))
        
        record_data_dict["record_index"] = ris_info_dict["index"]
    else: # noRIS case
        record_data_dict.update({
            "record_index": None, "ris_true_world_pos": None, "ris_positions_xyz_augmented_for_view": None,
            "rx_true_world_pos_for_steering": None, "rx_positions_xyz_augmented_for_view": None,
            "ris_true_world_orientation_rpy": None, "orientations_rpy_augmented_for_view": None
        })
    metadata_list_ref.append(record_data_dict)
    print(f"  Saved maps for {file_prefix_str}.")

def main():
    print("Starting dataset generation...")
    try:
        with open(JSON_INPUT_FILE, 'r') as f: data = json.load(f)
        tx_positions_all = np.load(TX_POS_FILE)
        city_map_original = cv2.imread(CITY_MAP_FILE)[:, :, 0]  # Load as grayscale
        if city_map_original is None: raise FileNotFoundError(f"{CITY_MAP_FILE} missing/invalid.")
        city_map_resized = cv2.resize(city_map_original, (CM_DIM, CM_DIM))
        print("Input files loaded.")
    except FileNotFoundError as e: print(f"Error loading input files: {e}"); return
    except Exception as e: print(f"An unexpected error during input loading: {e}"); return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    metadata_collector = []

    for tx_entry_item in data:
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

        # --- No RIS Case ---
        print("  Processing 'no RIS' cases...")
        # Extract unique RX positions from records
        unique_rx_positions = {}
        for i, record_item in enumerate(tx_entry_item["records"]):
            rx_pos = tuple(record_item["RX"])  # Use tuple as dict key
            if rx_pos not in unique_rx_positions:
                unique_rx_positions[rx_pos] = i
        
        # Generate no-RIS maps for each unique RX position
        for rx_idx, (rx_pos_tuple, record_idx) in enumerate(unique_rx_positions.items()):
            rx_pos_list = list(rx_pos_tuple)
            scene_no_ris = get_scene(SCENE_FILE)
            tx_obj_name_no_ris = f"tx{current_tx_id_str}_no_ris_rx{rx_idx}"
            transmitter_no_ris = Transmitter(tx_obj_name_no_ris, current_tx_pos_3d_scene + [0.0, 0.0, 2.0], [0.0, 0.0, 0.0])
            scene_no_ris.add(transmitter_no_ris)
            process_and_save_maps(scene_no_ris, current_tx_id_str, current_tx_pos, city_map_resized,
                                  tx_specific_output_folder, f"{current_tx_id_str}_noRIS_rx{rx_idx}", metadata_collector,
                                  is_ris_scenario=False, rx_center=rx_pos_list)

        # --- RIS Cases ---
        print(f"  Processing {len(tx_entry_item['records'])} RIS cases for TX_ID {current_tx_id_str}...")
        for i, record_item in enumerate(tx_entry_item["records"]):
            ris_position_3d = np.array(record_item["RIS"])
            rx_pos_for_steering_3d = np.array(record_item["RX"])
            current_ris_name = f"ris_{current_tx_id_str}_{i}"
            temp_rx_name_for_steering = f"rx_steer_{current_tx_id_str}_{i}"
            tx_obj_name_ris = f"tx{current_tx_id_str}_ris_case_{i}"

            print(f"    Record {i}: RIS at {ris_position_3d}, RX_steer at {rx_pos_for_steering_3d}")
            scene_with_ris = get_scene(SCENE_FILE)
            transmitter_ris_case = Transmitter(tx_obj_name_ris,  current_tx_pos_3d_scene + [0.0, 0.0, 2.0], [0.0, 0.0, 0.0])
            scene_with_ris.add(transmitter_ris_case)
            temp_rx_for_steering = Receiver(temp_rx_name_for_steering, rx_pos_for_steering_3d)
            scene_with_ris.add(temp_rx_for_steering)

            ris_object = RIS(name=current_ris_name, position=ris_position_3d,
                             orientation=RIS_INITIAL_ORIENTATION_EULER,
                             num_rows=RIS_NUM_ROWS, num_cols=RIS_NUM_COLS,
                             num_modes=RIS_NUM_MODES)
            ris_object.look_at((transmitter_ris_case.position + temp_rx_for_steering.position) / 2.0)
            scene_with_ris.add(ris_object)

            actual_ris_orientation_euler = RIS_INITIAL_ORIENTATION_EULER # Default if steering fails
             
            ris_object.phase_gradient_reflector(transmitter_ris_case.position, temp_rx_for_steering.position)
            actual_ris_orientation_euler = ris_object.orientation.numpy()
            
            ris_info_for_metadata = {
                "index": i, "ris_pos": ris_position_3d.tolist(),
                "rx_pos_for_steering": rx_pos_for_steering_3d.tolist(),
                "ris_actual_orientation_world_rpy": actual_ris_orientation_euler.tolist()
            }
            process_and_save_maps(scene_with_ris, current_tx_id_str, current_tx_pos, city_map_resized,
                                  tx_specific_output_folder, f"{current_tx_id_str}_RIS_{i}", metadata_collector,
                                  is_ris_scenario=True, ris_info_dict=ris_info_for_metadata)

    print(f"Saving metadata to {JSON_OUTPUT_FILE}...")
    try:
        with open(JSON_OUTPUT_FILE, 'w') as f: json.dump(metadata_collector, f, indent=2)
    except Exception as e: print(f"Error saving metadata JSON: {e}")
    print("Dataset generation complete.")

if __name__ == "__main__":
    try:
        # Import TensorFlow for potential use in steering if not already globally imported by Sionna
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus: print(f"TensorFlow - Available GPUs: {gpus}")
        else: print("TensorFlow - No GPUs found. Running on CPU.")
    except ImportError: print("TensorFlow not found, which might be an issue for Sionna.")
    except Exception as e: print(f"GPU check failed: {e}. CPU mode.")

    if not os.path.exists(ROOT): os.makedirs(ROOT)
    dummy_3d_path = os.path.join(ROOT, "USC_3D")
    if not os.path.exists(dummy_3d_path): os.makedirs(dummy_3d_path)
    if not os.path.exists(JSON_INPUT_FILE):
        print(f"Creating dummy '{JSON_INPUT_FILE}'...")
        with open(JSON_INPUT_FILE, 'w') as f: json.dump([{"tx_id":"0","records":[{"RIS":[-60,300,14],"RX":[45,250,14]}]}],f)
    if not os.path.exists(TX_POS_FILE):
        print(f"Creating dummy '{TX_POS_FILE}'...")
        np.save(TX_POS_FILE, np.array([[100,100],[-50,150]]))
    if not os.path.exists(CITY_MAP_FILE):
        print(f"Creating dummy '{CITY_MAP_FILE}'...")
        cv2.imwrite(CITY_MAP_FILE, np.full((CM_DIM,CM_DIM),128,dtype=np.uint8))
    scene_xml_path = os.path.join(dummy_3d_path, "USC.xml")
    if not os.path.exists(scene_xml_path):
        print(f"Creating dummy '{scene_xml_path}'...")
        with open(scene_xml_path, 'w') as f: f.write("""<?xml version="1.0"?><scene version="0.6.0"><integrator type="path"/><shape type="rectangle"><bsdf type="diffuse"><string name="radio_material" value="itu_very_dry_ground"/></bsdf></shape></scene>""")
    
    main()