import json
import math

INPUT_PATH = "metadata_final.json"
OUTPUT_PATH = "metadata_final.json"


def xyz_to_polar(x, y, z):
    """
    Convert XYZ position (relative to origin) to spherical polar representation:
    [rho, sin(theta), cos(theta), sin(phi), cos(phi)]
    where:
      rho   = sqrt(x^2 + y^2 + z^2)
      theta = atan2(y, x)          (azimuth)
      phi   = atan2(z, sqrt(x^2 + y^2))  (elevation)
    """
    rho = math.sqrt(x**2 + y**2 + z**2)
    theta = math.atan2(y, x)
    phi = math.atan2(z, math.sqrt(x**2 + y**2))
    return [rho, math.sin(theta), math.cos(theta), math.sin(phi), math.cos(phi)]


def xyz_to_rho_theta_phi(x, y, z):
    """Returns [rho, theta, phi] (raw angles in radians)."""
    rho = math.sqrt(x**2 + y**2 + z**2)
    theta = math.atan2(y, x)
    phi = math.atan2(z, math.sqrt(x**2 + y**2))
    return [rho, theta, phi]


with open(INPUT_PATH, "r") as f:
    data = json.load(f)

for entry in data:
    if entry.get("type") == "RIS":
        # True world position polar coords (sin/cos encoding)
        pos = entry.get("ris_true_world_pos")
        if pos is not None:
            entry["ris_true_world_pos_polar"] = xyz_to_polar(*pos)
            entry["ris_true_world_pos_rho_theta_phi"] = xyz_to_rho_theta_phi(*pos)
        else:
            entry["ris_true_world_pos_polar"] = None
            entry["ris_true_world_pos_rho_theta_phi"] = None

        # Augmented view polar coords
        aug_positions = entry.get("ris_positions_xyz_augmented_for_view")
        if aug_positions is not None:
            entry["ris_positions_polar_augmented_for_view"] = [
                xyz_to_polar(*p) for p in aug_positions
            ]
            entry["ris_positions_rho_theta_phi_augmented_for_view"] = [
                xyz_to_rho_theta_phi(*p) for p in aug_positions
            ]
        else:
            entry["ris_positions_polar_augmented_for_view"] = None
            entry["ris_positions_rho_theta_phi_augmented_for_view"] = None
    else:
        # noRIS entries get null for all fields
        entry["ris_true_world_pos_polar"] = None
        entry["ris_positions_polar_augmented_for_view"] = None
        entry["ris_true_world_pos_rho_theta_phi"] = None
        entry["ris_positions_rho_theta_phi_augmented_for_view"] = None

with open(OUTPUT_PATH, "w") as f:
    json.dump(data, f, indent=2)

print(f"Done. Processed {len(data)} entries.")

# Print a sample RIS entry to verify
for entry in data:
    if entry.get("type") == "RIS":
        print("\nSample RIS entry:")
        print("  ris_true_world_pos:", entry["ris_true_world_pos"])
        print("  ris_true_world_pos_polar:", entry["ris_true_world_pos_polar"])
        print("  ris_positions_polar_augmented_for_view (first):",
              entry["ris_positions_polar_augmented_for_view"][0]
              if entry["ris_positions_polar_augmented_for_view"] else None)
        break
