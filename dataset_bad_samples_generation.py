import sionna

# Import Sionna RT components
import numpy as np
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera, RIS
import time
from tqdm import tqdm

TX = np.load("Data/tx_positions.npy")

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

def get_no_ris_coverage_map(tx_coordinates, center=None, size=None):
    scene = get_scene("DataRIS/USC_3D/USC.xml")
    tx = Transmitter(f"tx", tx_coordinates+[0, 0, 2], [0.0, 0.0, 0.0])
    scene.add(tx)
    orientation = None if center is None and size is None else [0.0, 0.0, 0.0]
    cm_no_ris = scene.coverage_map(
            cm_cell_size=(5.0, 5.0), ris=False, cm_center=center, cm_size=size, cm_orientation=orientation,
            diffraction=True, scattering=True, edge_diffraction=True, max_depth=1000, num_samples=2*(10**6)
        )
    return cm_no_ris

def get_ris_coverage_map(tx_coordinates, ris_coordinates, rx_coordinates):
    scene = get_scene("DataRIS/USC_3D/USC.xml")
    tx = Transmitter(f"tx", tx_coordinates+[0, 0, 2], [0.0, 0.0, 0.0])
    scene.add(tx)

    rx = Receiver("rx", position=rx_coordinates)

    ris = RIS(name="ris",
              position=ris_coordinates,
              num_rows=32,
              num_cols=32,
              num_modes=1,
              orientation=[-np.pi/2, 0, 0])
    ris.look_at((tx.position + rx.position) / 2)
    scene.add(ris)
    ris.phase_gradient_reflector(tx.position, rx.position)
    cm_ris = scene.coverage_map(
            cm_cell_size=(5.0, 5.0), ris=True, cm_center=rx_coordinates, cm_size=[50, 50], cm_orientation=[0, 0, 0],
            diffraction=True, scattering=True, edge_diffraction=True, max_depth=1000, num_samples=2*(10**6)
        )
    return cm_ris


def generate_ris_circles(center_pos_xy, ris_radii, ris_angles, ris_z):
    """Generates a list of RIS candidate positions in circles."""
    ris_candidates = []
    for radius in ris_radii:
        for angle in ris_angles:
            ris_x = center_pos_xy[0] + radius * np.cos(angle)
            ris_y = center_pos_xy[1] + radius * np.sin(angle)

            ris_pos_xy = np.array([ris_x, ris_y])
            if is_within_map(ris_pos_xy, MAP_MIN_XY, MAP_MAX_XY):
                ris_candidates.append(np.array([ris_x, ris_y, ris_z]))
    return ris_candidates


def get_gain_map(cm_no_ris, cm_ris):
    return 10 * np.log10(cm_ris.path_gain[0] / (cm_no_ris.path_gain[0] + 1e-20))


def check_patch_condition_bad(ris_gain_map, threshold_db=20, percent_required=0.2):
    """Checks if < percent_required of pixels are >= threshold_db (i.e., a bad sample)."""
    total_pixels = ris_gain_map.size
    valid_pixels = np.sum(ris_gain_map >= threshold_db)
    return (valid_pixels / total_pixels) < percent_required


MAP_MIN_XY = -450.0
MAP_MAX_XY = 450.0

def is_within_map(point_xy, min_val, max_val):
    """Checks if the (x, y) coordinates are within the map boundaries."""
    x, y = point_xy
    return (min_val <= x <= max_val) and (min_val <= y <= max_val)


# --- Global parameters ---
NUM_RX_SAMPLES = 30

# --- Angle sectors for RIS search ---
angles_sector_1 = np.arange(-30, 31, 10)
angles_sector_2 = np.arange(150, 211, 10)
ris_angles_deg = np.concatenate([angles_sector_1, angles_sector_2])
ris_angles = np.radians(ris_angles_deg)

# --- Main loop over all TX positions ---
all_found_positions = []
total_point_counter = 0
start_time = time.time()

print(f"Starting bad-sample search over {len(TX)} TX positions...")
print("=" * 60)

for tx_idx, tx_position in enumerate(TX):
    print(f"\n[TX {tx_idx}/{len(TX)-1}] Position: {tx_position}")
    print("-" * 50)

    ris_z = tx_position[2] + 2  # RIS z-coord fixed to TX's z-coord + 2

    # Compute global no-RIS coverage map for this TX
    path_gain_no_ris_map = get_no_ris_coverage_map(tx_position)
    rx_positions = path_gain_no_ris_map.sample_positions(
        NUM_RX_SAMPLES, max_val_db=-100, max_dist=300
    )[0][0]

    print(f"  Sampled {len(rx_positions)} RX positions.")
    print("  Looking for 10x10 patches with NO pixel having gain > 20dB...")

    found_for_this_tx = False

    for rx_position in tqdm(rx_positions, desc=f"  TX {tx_idx} RX loop"):
        if found_for_this_tx:
            break

        rx_pos_xy = rx_position[:2]
        tx_pos_xy = tx_position[:2]

        d_tx_rx_2d = np.linalg.norm(tx_pos_xy - rx_pos_xy)
        max_tx_radius = 0.5 * d_tx_rx_2d
        ris_radii = np.arange(50, max(51, max_tx_radius), 10)

        ris_candidates_around_tx = generate_ris_circles(tx_pos_xy, ris_radii, ris_angles, ris_z)
        all_ris_to_check = ris_candidates_around_tx

        print(f"    Generated {len(all_ris_to_check)} RIS candidates for this RX.")

        path_gain_no_ris_local = get_no_ris_coverage_map(
            tx_position, center=rx_position, size=[50, 50]
        )

        for ris_candidate in tqdm(all_ris_to_check, desc="    RIS candidates", leave=False):
            total_point_counter += 1

            path_gain_ris_map = get_ris_coverage_map(tx_position, ris_candidate, rx_position)
            ris_gain_db_map = get_gain_map(path_gain_no_ris_local, path_gain_ris_map)

            if check_patch_condition_bad(ris_gain_db_map, threshold_db=20, percent_required=0):
                print(f"\n  ❌ Found bad patch at combo #{total_point_counter}!")
                position_data = {
                    'tx_idx': tx_idx,
                    'tx_pos': tx_position,
                    'ris_pos': ris_candidate,
                    'rx_pos': rx_position.numpy(),
                }
                all_found_positions.append(position_data)

                print(f"     TX:  {np.array2string(tx_position)}")
                print(f"     RIS: {np.array2string(ris_candidate, precision=2)}")
                print(f"     RX:  {np.array2string(rx_position.numpy(), precision=2)}")

                found_for_this_tx = True
                break  # Move to next TX once one bad patch is found

    if found_for_this_tx:
        print(f"  --> Bad sample found for TX {tx_idx}. Moving to next TX.")
    else:
        print(f"  --> No bad sample found for TX {tx_idx} after exhaustive search.")

# --- Summary ---
end_time = time.time()
print("\n" + "=" * 60)
print(f"Search Complete in {end_time - start_time:.2f} seconds.")
print(f"Checked {total_point_counter} total (TX, RIS, RX) combinations.")
print(f"Found {len(all_found_positions)} bad samples across {len(TX)} TX positions.")
print("=" * 60)

for k, pos_data in enumerate(all_found_positions):
    print(f"\n--- Bad Sample {k+1} (TX index {pos_data['tx_idx']}) ---")
    print(f"  TX:  {np.array2string(pos_data['tx_pos'])}")
    print(f"  RIS: {np.array2string(pos_data['ris_pos'], precision=2)}")
    print(f"  RX:  {np.array2string(pos_data['rx_pos'], precision=2)}")
