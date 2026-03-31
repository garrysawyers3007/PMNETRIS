import os
import json
import numpy as np

# --- Constants (match preprocessRIS.py) ---
OUTPUT_ROOT = "datasetGeoRIS"
JSON_INPUT_FILE = "dataset_new_32x32.json"
METADATA_POLAR_FILE = os.path.join(OUTPUT_ROOT, "metadata_polar.json")
JSON_OUTPUT_FILE = os.path.join(OUTPUT_ROOT, "metadata.json")

MAP_TYPES = ["tx_map", "rx_map", "ris_map", "power_map", "city_map"]
NUM_VIEWS = 4


def augment_position_ordered(world_xyz_array):
    """
    Transforms a true world 3D coordinate [x, y, z] to correspond with
    the four augmented 2D image views. Matches logic in preprocessRIS.py.
    """
    x, y, z = world_xyz_array
    return [
        [x,  y, z],   # view 0: original
        [x, -y, z],   # view 1: vertical flip
        [-x, y, z],   # view 2: horizontal flip
        [-x, -y, z],  # view 3: both flips
    ]


def build_paths(output_folder, prefix):
    """Build {map_type: [path_v0, ..., path_v3]} for a given image file prefix."""
    return {
        map_type: [
            os.path.abspath(os.path.join(output_folder, f"{prefix}_{map_type}_{v}.png"))
            for v in range(NUM_VIEWS)
        ]
        for map_type in MAP_TYPES
    }


def build_orientation_lookup(metadata_polar):
    """
    Build lookup: {(tx_id_str, record_index_int): orientation + polar fields}
    from the RIS entries of metadata_polar.json.
    """
    lookup = {}
    for entry in metadata_polar:
        if entry.get("type") != "RIS":
            continue
        key = (entry["tx_id"], entry["record_index"])
        lookup[key] = {
            "ris_true_world_orientation_rpy":           entry.get("ris_true_world_orientation_rpy"),
            "orientations_rpy_augmented_for_view":      entry.get("orientations_rpy_augmented_for_view"),
            "ris_true_world_pos_polar":                 entry.get("ris_true_world_pos_polar"),
            "ris_positions_polar_augmented_for_view":   entry.get("ris_positions_polar_augmented_for_view"),
            "ris_true_world_pos_rho_theta_phi":         entry.get("ris_true_world_pos_rho_theta_phi"),
            "ris_positions_rho_theta_phi_augmented_for_view": entry.get("ris_positions_rho_theta_phi_augmented_for_view"),
        }
    return lookup


def main():
    print("Loading input files...")
    with open(JSON_INPUT_FILE, "r") as f:
        dataset = json.load(f)
    with open(METADATA_POLAR_FILE, "r") as f:
        metadata_polar = json.load(f)

    orientation_lookup = build_orientation_lookup(metadata_polar)
    print(f"Loaded orientation data for {len(orientation_lookup)} RIS records from metadata_polar.json.")

    metadata = []

    for tx_entry in dataset:
        tx_id = tx_entry["tx_id"]
        records = tx_entry["records"]
        output_folder = os.path.join(OUTPUT_ROOT, tx_id)
        print(f"Processing TX_ID: {tx_id} ({len(records)} RIS records)...")

        # --- Build unique RX → rx_idx mapping (order of first appearance) ---
        unique_rx_map = {}  # {rx_pos_tuple: rx_idx}
        for record in records:
            rx_tuple = tuple(record["RX"])
            if rx_tuple not in unique_rx_map:
                unique_rx_map[rx_tuple] = len(unique_rx_map)

        # --- noRIS records (one per unique RX position) ---
        for rx_tuple, rx_idx in unique_rx_map.items():
            prefix = f"{tx_id}_noRIS_rx{rx_idx}"
            paths = build_paths(output_folder, prefix)

            metadata.append({
                "tx_id":   tx_id,
                "type":    "noRIS",
                "paths":   paths,
                "record_index": None,
                "ris_true_world_pos":                        None,
                "ris_positions_xyz_augmented_for_view":      None,
                "rx_true_world_pos_for_steering":            None,
                "rx_positions_xyz_augmented_for_view":       None,
                "ris_true_world_orientation_rpy":            None,
                "orientations_rpy_augmented_for_view":       None,
                "ris_true_world_pos_polar":                  None,
                "ris_positions_polar_augmented_for_view":    None,
                "ris_true_world_pos_rho_theta_phi":          None,
                "ris_positions_rho_theta_phi_augmented_for_view": None,
                # For noRIS, the baseline power map is its own power map
                "noRIS_power_map": paths["power_map"],
            })

        # --- RIS records ---
        for i, record_item in enumerate(records):
            ris_pos = record_item["RIS"]
            rx_pos  = record_item["RX"]
            rx_idx  = unique_rx_map[tuple(rx_pos)]

            prefix = f"{tx_id}_RIS_{i}"
            paths  = build_paths(output_folder, prefix)

            # Corresponding noRIS power map (same TX, same RX position)
            noris_prefix = f"{tx_id}_noRIS_rx{rx_idx}"
            noris_power_map = [
                os.path.abspath(os.path.join(output_folder, f"{noris_prefix}_power_map_{v}.png"))
                for v in range(NUM_VIEWS)
            ]

            # Orientation + polar fields from metadata_polar.json
            orient = orientation_lookup.get((tx_id, i), {})

            metadata.append({
                "tx_id":   tx_id,
                "type":    "RIS",
                "paths":   paths,
                "record_index": i,
                "ris_true_world_pos":                   ris_pos,
                "ris_positions_xyz_augmented_for_view": augment_position_ordered(np.array(ris_pos)),
                "rx_true_world_pos_for_steering":       rx_pos,
                "rx_positions_xyz_augmented_for_view":  augment_position_ordered(np.array(rx_pos)),
                "ris_true_world_orientation_rpy":                    orient.get("ris_true_world_orientation_rpy"),
                "orientations_rpy_augmented_for_view":               orient.get("orientations_rpy_augmented_for_view"),
                "ris_true_world_pos_polar":                          orient.get("ris_true_world_pos_polar"),
                "ris_positions_polar_augmented_for_view":            orient.get("ris_positions_polar_augmented_for_view"),
                "ris_true_world_pos_rho_theta_phi":                  orient.get("ris_true_world_pos_rho_theta_phi"),
                "ris_positions_rho_theta_phi_augmented_for_view":    orient.get("ris_positions_rho_theta_phi_augmented_for_view"),
                # Corresponding noRIS power map for this TX + RX position
                "noRIS_power_map": noris_power_map,
            })

    n_noris = sum(1 for r in metadata if r["type"] == "noRIS")
    n_ris   = sum(1 for r in metadata if r["type"] == "RIS")
    print(f"\nGenerated {len(metadata)} records total: {n_noris} noRIS, {n_ris} RIS.")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(JSON_OUTPUT_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {JSON_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
