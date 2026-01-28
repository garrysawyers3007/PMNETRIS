import sionna

# Import Sionna RT components
import numpy as np
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera, RIS
import time
from scipy import signal
from tqdm import tqdm
import argparse
import sys

TX =  np.load("Data/tx_positions.npy")

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
            cm_cell_size=(5.0, 5.0), ris=False, cm_center=center, cm_size=size, cm_orientation = orientation,
            diffraction=True, scattering=True, edge_diffraction=True, max_depth = 1000, num_samples = 2*(10**6)
        )
    return cm_no_ris

def get_ris_coverage_map(tx_coordinates, ris_coordinates, rx_coordinates):
    scene = get_scene("DataRIS/USC_3D/USC.xml")
    tx = Transmitter(f"tx", tx_coordinates+[0, 0, 2], [0.0, 0.0, 0.0])
    scene.add(tx) 
    
    rx = Receiver("rx", position=rx_coordinates)
    
    ris = RIS(name="ris",
              position = ris_coordinates,
              num_rows=32,
              num_cols=32,
              num_modes=1,
              orientation = [-np.pi/2,0,0])
    ris.look_at((tx.position+rx.position)/2)
    # ris.look_at(tx.position)
    scene.add(ris)
    ris.phase_gradient_reflector(tx.position, rx.position)
    cm_ris = scene.coverage_map(
            cm_cell_size=(5.0, 5.0), ris=True, cm_center=rx_coordinates, cm_size=[50, 50], cm_orientation = [0,0,0],
            diffraction=True, scattering=True, edge_diffraction=True, max_depth = 1000, num_samples = 2*(10**6)
        )
    return cm_ris


def generate_ris_circles(center_pos_xy, ris_radii, ris_angles, ris_z):
    """Generates a list of RIS candidate positions in circles."""
    ris_candidates = []
    for radius in ris_radii:
        for angle in ris_angles:
            ris_x = center_pos_xy[0] + radius * np.cos(angle)
            ris_y = center_pos_xy[1] + radius * np.sin(angle)
            
            # Check map boundary right here
            ris_pos_xy = np.array([ris_x, ris_y])
            if is_within_map(ris_pos_xy, MAP_MIN_XY, MAP_MAX_XY):
                ris_candidates.append(np.array([ris_x, ris_y, ris_z]))
    return ris_candidates


def get_gain_map(cm_no_ris, cm_ris):
    return 10*np.log10(cm_ris.path_gain[0]/(cm_no_ris.path_gain[0]+1e-20))

def check_patch_condition(ris_gain_map, patch_size=10, threshold_db=20, percent_required=0.8):
    # 1. Create a binary map: 1 where gain > threshold, 0 otherwise.
    binary_map = (ris_gain_map > threshold_db).astype(int)

    # 2. Define the number of pixels required in a patch
    pixels_per_patch = patch_size * patch_size
    pixels_required = int(pixels_per_patch * percent_required) # e.g., 80

    # 3. Use 2D convolution to sum the '1's in every possible
    #    (patch_size x patch_size) window. This is extremely fast.
    #    The 'kernel' is just a 10x10 matrix of ones.
    kernel = np.ones((patch_size, patch_size))
    
    # 'mode=valid' means we only compute sums for *full* 10x10 patches.
    # The result is a (180-10+1) x (180-10+1) = 171x171 map
    # where each value is the *count* of pixels > 20dB in that patch.
    summed_patches = signal.convolve2d(binary_map, kernel, mode='valid')

    # 4. Check if *any* value in the resulting map meets our criteria
    #    (e.g., if any patch has a sum >= 80)
    if np.any(summed_patches >= pixels_required):
        return True
    else:
        return False
    
def check_coverage_condition(ris_gain_map, threshold_db=20, percent_required=0.2):
    """Checks if >= percent_required of pixels are >= threshold_db."""
    total_pixels = ris_gain_map.size
    valid_pixels = np.sum(ris_gain_map >= threshold_db)
    return (valid_pixels / total_pixels) >= percent_required
    

MAP_MIN_XY = -450.0
MAP_MAX_XY = 450.0

# --- NEW: Helper function for the boundary check ---
def is_within_map(point_xy, min_val, max_val):
    """Checks if the (x, y) coordinates are within the map boundaries."""
    # point_xy is expected to be a 2D array/tuple like (x, y)
    x, y = point_xy
    return (min_val <= x <= max_val) and (min_val <= y <= max_val)

# --- End of new additions ---

# --- Command Line Argument Parsing ---
parser = argparse.ArgumentParser(description='Generate RIS coverage data for a specific TX index.')
parser.add_argument('index', type=int, help='Index of the TX position to use (0 to N-1)')
args = parser.parse_args()

i = args.index

# Basic validation
if i < 0 or i >= len(TX):
    print(f"Error: Index {i} is out of bounds. TX array has {len(TX)} elements.")
    sys.exit(1)
# -------------------------------------

# i = 8
tx_position = TX[i] # e.g., np.array([0, 0, 10])

print(f"Using TX position at index {i}: {tx_position}")
NUM_POINTS = 12 # Total valid points to find
POINTS_PER_RX_LIMIT = 3 # Max points to find for a single RX

# --- Define parameters for RIS circles ---
ris_radii = np.arange(50, 151, 10) # 50, 60, ..., 150
angles_sector_1 = np.arange(-30, 31, 10) 
# Sector 2: 150 to 210 degrees (Back) -> [150, 160, ..., 210]
angles_sector_2 = np.arange(150, 211, 10)

# Combine and convert to radians
ris_angles_deg = np.concatenate([angles_sector_1, angles_sector_2])
ris_angles = np.radians(ris_angles_deg)

# --- Define fixed Z-coordinates (YOUR NEW IDEA) ---
ris_z = tx_position[2] + 2  # RIS z-coord is fixed to TX's z-coord
# --- End of new additions ---

# 4. Perform the search
found_positions = []
point_counter = 0

path_gain_no_ris_map = get_no_ris_coverage_map(tx_position)
rx_positions = path_gain_no_ris_map.sample_positions(30, max_val_db=-100, max_dist=300)[0][0]
print(f"  (Checking RIS circles around TX and each RX)")
print("-" * 50)
print("Looking for 10x10 patches where 20% of pixels > 20dB...")
start_time = time.time()

# --- Pre-generate the RIS points around the TX ---
tx_pos_xy = tx_position[:2]

# print(f"Generated {len(ris_candidates_around_tx)} RIS candidates around TX.")

# --- Loop 1: Random RX Samples ---
for rx_position in tqdm(rx_positions):

    # Initialize counter for this specific RX
    valid_points_for_this_rx = 0
    # 1. Generate a random RX position
    rx_pos_xy = rx_position[:2]

    d_tx_rx_2d = np.linalg.norm(tx_pos_xy - rx_pos_xy)
    max_tx_radius = 0.5 * d_tx_rx_2d
    ris_radii = np.arange(50, max_tx_radius, 10)
    ris_candidates_around_tx = generate_ris_circles(tx_pos_xy, ris_radii, ris_angles, ris_z)
    # 2. Generate RIS points in circles *around this RX*
    ris_candidates_around_rx = generate_ris_circles(rx_pos_xy, ris_radii, ris_angles, ris_z)
    
    # Combine the two lists of RIS candidates
    all_ris_to_check = ris_candidates_around_tx #+ ris_candidates_around_rx
    print(f"Generated {len(all_ris_to_check)} RIS candiates")
    # if rx_sample_num % 10 == 0:
    #     print(f"Checking RX Sample #{rx_sample_num+1}/{NUM_RX_SAMPLES} (Checking {len(all_ris_to_check)} RIS points for it)")

    path_gain_no_ris_local = get_no_ris_coverage_map(tx_position, center=rx_position, size=[50, 50])

    # --- Loop 2: RIS Candidates ---
    for ris_candidate in tqdm(all_ris_to_check):
        
        point_counter += 1 # A "point" is one (tx, rx, ris) combo

        # --- Main Calculation ---
        path_gain_ris_map = get_ris_coverage_map(tx_position, ris_candidate, rx_position)
        
        ris_gain_db_map = get_gain_map(path_gain_no_ris_local, path_gain_ris_map)
        
        # --- Check Condition ---
        if check_coverage_condition(ris_gain_db_map, threshold_db=20, percent_required=0.15):
            print(f"\n✅ Found valid patch at combo #{point_counter}!")
            position_data = {
                'tx_pos': tx_position,
                'ris_pos': ris_candidate,
                'rx_pos': rx_position.numpy(),
            }
            found_positions.append(position_data)

            # Increment the counter for this RX
            valid_points_for_this_rx += 1

            print(f"   TX: {np.array2string(tx_position)}")
            print(f"   RIS: {np.array2string(ris_candidate, precision=2)}")
            print(f"   RX: {np.array2string(rx_position.numpy(), precision=2)}\n")
            
            if len(found_positions) >= NUM_POINTS:
                print(f"Found {NUM_POINTS} valid points, stopping search.")
                break

            if valid_points_for_this_rx >= POINTS_PER_RX_LIMIT:
                print(f"--> Reached limit of {POINTS_PER_RX_LIMIT} points for this RX. Moving to next RX...")
                break 
                
    if len(found_positions) >= NUM_POINTS:
        break
# --- End RX Loop ---

end_time = time.time()
print("-" * 50)
print(f"Search Complete in {end_time - start_time:.2f} seconds.")
print(f"Checked {point_counter} total (TX, RIS, RX) combinations.")
print(f"Found {len(found_positions)} valid combinations.")

for i, pos_data in enumerate(found_positions):
    print(f"\n--- Position {i+1} ---")
    print(f"  TX: {np.array2string(pos_data['tx_pos'])}")
    print(f"  RIS: {np.array2string(pos_data['ris_pos'], precision=2)}")
    print(f"  RX: {np.array2string(pos_data['rx_pos'], precision=2)}")