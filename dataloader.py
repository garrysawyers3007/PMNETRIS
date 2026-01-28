
from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import json
import warnings

class PMnet_data_usc(Dataset):
    def __init__(self,
                 dir_dataset="", # Path to the root directory containing metadata.json (e.g., "datasetRIS/")
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Resize((256, 256), antialias=True) 
                 ]),
                 # IMPORTANT: Define these bounds based on your actual dataset's RIS coordinates
                 # Example: ris_pos_min=[-400, -200, 0], ris_pos_max=[300, 400, 70]
                 ris_pos_min=None, 
                 ris_pos_max=None,
                 get_paths = False):
        
        self.dir_dataset = dir_dataset
        self.transform = transform
        self.metadata_path = os.path.join(self.dir_dataset, "metadata.json")
        self.get_paths = get_paths
        
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"metadata.json not found at {self.metadata_path}")

        with open(self.metadata_path, 'r') as f:
            original_metadata_records = json.load(f)

        self.ris_pos_min = np.array(ris_pos_min, dtype=np.float32) if ris_pos_min else None
        self.ris_pos_max = np.array(ris_pos_max, dtype=np.float32) if ris_pos_max else None
        self.ris_pos_range = self.ris_pos_max - self.ris_pos_min if (self.ris_pos_min is not None and self.ris_pos_max is not None) else None

        if ris_pos_min is None or ris_pos_max is None:
            print("Warning: RIS position normalization bounds (ris_pos_min, ris_pos_max) not provided. RIS positions will not be normalized.")
            print("It's highly recommended to calculate these from your dataset or provide them for proper normalization.")
        elif not (self.ris_pos_min.shape == (3,) and self.ris_pos_max.shape == (3,)):
            raise ValueError("ris_pos_min and ris_pos_max must be 3-element arrays/lists (X,Y,Z).")
        elif np.any(self.ris_pos_min >= self.ris_pos_max):
            print("Warning: Some elements in ris_pos_min are not strictly less than ris_pos_max. Normalization might behave unexpectedly.")

        self.metadata_records = self._process_metadata(original_metadata_records)

    def __len__(self):
        return len(self.metadata_records)

    def _normalize_pos(self, pos_np, min_bounds, range_bounds):
        """Normalizes a 3D position vector to the range [0, 1]."""
        return (pos_np - min_bounds) / range_bounds
    
    def _process_metadata(self, original_records):
        processed_list = []
        for record_idx, record in enumerate(original_records):
            paths_dict = record.get("paths")
            if not isinstance(paths_dict, dict) or \
               not all(k in paths_dict for k in ["city_map", "tx_map", "power_map"]):
                print(f"Warning: Record {record_idx} has missing or malformed 'paths' dictionary. Skipping this record.")
                continue

            num_augmentations = 0
            if paths_dict["city_map"]: # Check if the list is not empty
                 num_augmentations = len(paths_dict["city_map"])
            
            if num_augmentations == 0:
                print(f"Warning: Record {record_idx} has no image paths. Skipping this record.")
                continue


            for i in range(num_augmentations): # Typically 0, 1, 2, 3
                if not (len(paths_dict["tx_map"]) > i and len(paths_dict["power_map"]) > i):
                    print(f"Warning: Missing some map paths for augmentation index {i} in record {record_idx}. Skipping this augmentation.")
                    continue

                data_point = {
                    "tx_id": record.get("tx_id"),
                    "type": record.get("type"),
                    "city_map_path": paths_dict["city_map"][i],
                    "tx_map_path": paths_dict["tx_map"][i],
                    "power_map_path": paths_dict["power_map"][i],
                    "is_ris_present_flag": 0.0,
                    "ris_pos_normalized": np.zeros(3, dtype=np.float32), # Default for noRIS
                    "ris_orientation_rpy_for_view": np.zeros(3, dtype=np.float32), # Default for noRIS
                    "rx_pos_normalized": np.zeros(3, dtype=np.float32) # Default for noRIS
                }

                if record.get("type") == "RIS":
                    augmented_positions = record.get("ris_positions_xyz_augmented_for_view")
                    augmented_rx_positions = record.get("rx_positions_xyz_augmented_for_view")
                    augmented_orientations = record.get("orientations_rpy_augmented_for_view")

                    if isinstance(augmented_positions, list) and len(augmented_positions) > i and \
                        isinstance(augmented_orientations, list) and len(augmented_orientations) > i:
                        
                        specific_ris_pos = augmented_positions[i]
                        specific_rx_pos = augmented_rx_positions[i]
                        specific_orientation = augmented_orientations[i]
                        
                        # Ensure the specific augmented orientation is a list of 3 floats
                        if isinstance(specific_ris_pos, list) and len(specific_ris_pos) == 3 and \
                           isinstance(specific_rx_pos, list) and len(specific_rx_pos) == 3 and \
                           isinstance(specific_orientation, list) and len(specific_orientation) == 3:
                        
                            data_point["ris_pos_normalized"] = self._normalize_pos(np.array(specific_ris_pos, dtype=np.float32), self.ris_pos_min, self.ris_pos_range)
                            data_point["rx_pos_normalized"] = self._normalize_pos(np.array(specific_rx_pos, dtype=np.float32), self.ris_pos_min, self.ris_pos_range)
                            data_point["ris_orientation_rpy_for_view"] = np.array(specific_orientation, dtype=np.float32)
                            
                            data_point["is_ris_present_flag"] = 1.0
                        else:
                            print(f"Warning: Malformed position/orientation for view {i} in RIS record {record_idx}. Treating as noRIS.")

                        data_point["is_ris_present_flag"] = 1.0
                    else:
                        print(f"Warning: Missing or malformed RIS data for view {i} in record {record_idx}. Treating as noRIS for this augmentation.")
                        # Defaults for noRIS will be used.
                
                processed_list.append(data_point)
        return processed_list

    def __getitem__(self, idx):
        data_point = self.metadata_records[idx]
        
        try:
            image_buildings = np.asarray(io.imread(data_point["city_map_path"])) 
            if image_buildings.ndim == 3 and image_buildings.shape[-1] >= 3:
                image_buildings = image_buildings[:,:,0]

            image_tx = np.asarray(io.imread(data_point["tx_map_path"])) 
            if image_tx.ndim == 3 and image_tx.shape[-1] >= 3:
                image_tx = image_tx[:,:,0]

            image_power = np.asarray(io.imread(data_point["power_map_path"]))
            if image_power.ndim == 3 and image_power.shape[-1] >= 3:
                image_power = image_power[:,:,0]

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not load image for data_point {idx} (path: {e.filename}). Error: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while loading images for data_point {idx}: {e}")

        if image_buildings.ndim != 2: raise ValueError(f"Building map for data_point {idx} is not 2D after processing.")
        if image_tx.ndim != 2: raise ValueError(f"TX map for data_point {idx} is not 2D after processing.")
        
        inputs_np = np.stack([image_buildings, image_tx], axis=-1) 

        # Construct ris_info_tensor from pre-processed data_point fields
        ris_info_list = list(data_point["ris_pos_normalized"]) + \
                        list(data_point["ris_orientation_rpy_for_view"]) + \
                        list(data_point["rx_pos_normalized"]) + \
                        [data_point["is_ris_present_flag"]]
        ris_info_tensor = torch.tensor(ris_info_list, dtype=torch.float32)
        
        inputs_tensor = inputs_np 
        power_tensor_np = image_power 

        if self.transform:
            inputs_tensor = self.transform(inputs_np).type(torch.float32)
            
            if power_tensor_np.dtype == np.uint8:
                power_tensor = self.transform(power_tensor_np).type(torch.float32)
            else: 
                power_tensor = self.transform(power_tensor_np.astype(np.uint8)).type(torch.float32)

        if self.get_paths:
            return inputs_tensor, ris_info_tensor, power_tensor, data_point["city_map_path"], data_point["tx_map_path"], data_point["power_map_path"]
            
        return inputs_tensor, ris_info_tensor, power_tensor
    
# if __name__ == "__main__":
#     # Example usage
#     dataset = PMnet_data_usc(
#         dir_dataset="datasetRIS/",
#         ris_pos_min=[-400, -400, 0],
#         ris_pos_max=[300, 400, 70]
#     )
    
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
#     len_dataset = len(dataset)
#     for inputs, ris_params, targets in dataloader:
#         print("Inputs shape:", inputs.shape)
#         print("RIS Params shape:", ris_params.shape)
#         print("Targets shape:", targets.shape)
#         print("Length of dataset:", len_dataset)
#         break  # Just to test the first batch
