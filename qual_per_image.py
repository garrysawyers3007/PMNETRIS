import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import pandas as pd
from dataloader import PMnet_data_usc
from network.pmnet import PMNetFiLM
from loss import L1_loss, MSE, RMSE, roi_rmse_loss
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def test_per_image(model, dataloader, device):
    """
    Iterates through a dataloader, calculates RMSE for each individual sample,
    and returns a list of results.
    """
    model.eval()
    results = []
    
    pbar = tqdm(dataloader, desc="Calculating Per-Image RMSE")
    
    with torch.no_grad():
        for batch_idx, (inputs, ris_params, targets, city_path, tx_path, power_path) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            ris_params = ris_params.to(device)

            preds = model(inputs, ris_params)
            preds = torch.clip(preds, 0, 1)

            for i in range(inputs.shape[0]):
                pred_single = preds[i]
                target_single = targets[i]
                c = city_path[i]
                t = tx_path[i]
                p = power_path[i]
                loss = RMSE(pred_single.unsqueeze(0), target_single.unsqueeze(0))
                rmse_value = loss.item()
                
                sample_index = batch_idx * dataloader.batch_size + i
                
                results.append({
                    "sample_index": sample_index,
                    "city_map_path": c,
                    "tx_map_path": t,
                    "power_map_path": p,
                    "ris_true_world_pos": ris_params[i][:3].cpu().numpy().tolist(),
                    "ris_orientation_rpy": ris_params[i][3:6].cpu().numpy().tolist(),
                    "rx_true_world_pos": ris_params[i][6:9].cpu().numpy().tolist(),
                    "is_ris_present": int(ris_params[i][9].item()),
                    "rmse": rmse_value
                })

    return results

# =================================================================================
# MAIN SCRIPT EXECUTION
# =================================================================================
if __name__ == '__main__':
    # --- Configuration ---
    DATASET_DIR = "datasetRISNewCorrected"
    MODEL_PATH = "datasetRISNewCorrected/PMNet_results/augmented_config_USC_pmnetV3_V2_epoch100/8_0.0001_0.45_10/model_0.03808.pt"
    
    RIS_POS_MIN = [-400.0, -400.0, 0.0] 
    RIS_POS_MAX = [400.0, 400.0, 55.0]

    BATCH_SIZE = 16
    set_seed(1234)
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        dataset = PMnet_data_usc(dir_dataset=DATASET_DIR, 
                                       ris_pos_min=RIS_POS_MIN,
                                       ris_pos_max=RIS_POS_MAX,
                                       get_paths=True)
        
        dataset_size = len(dataset)

        train_size = int(dataset_size * 0.9)
        # validation_size = int(dataset_size * 0.1)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print(f"Loaded dataset from '{DATASET_DIR}' with {len(test_dataset)} samples.")
    except FileNotFoundError as e:
        print(f"Error: Could not find dataset files. {e}")
        exit()

    # Initialize Model
    model = PMNetFiLM(
        n_blocks=[3, 3, 27, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=8,
        cond_features=10).to(device) 
    
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model weights not found at '{MODEL_PATH}'.")
        print("A new dummy model will be used for this run, which will produce random results.")
    else:
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"Successfully loaded model weights from '{MODEL_PATH}'.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            exit()

    # --- Run Evaluation ---
    rmse_results = test_per_image(model, test_loader, device)

    # --- Report Results ---
    if rmse_results:
        results_df = pd.DataFrame(rmse_results)
        
        mean_rmse = results_df['rmse'].mean()
        std_rmse = results_df['rmse'].std()
        min_rmse = results_df['rmse'].min()
        max_rmse = results_df['rmse'].max()
        
        print("\n--- Evaluation Summary ---")
        print(f"Mean RMSE:   {mean_rmse:.6f}")
        print(f"Std Dev RMSE:{std_rmse:.6f}")
        print(f"Min RMSE:    {min_rmse:.6f}")
        print(f"Max RMSE:    {max_rmse:.6f}")
        print("--------------------------\n")
        
        output_csv_path = "test_rmse_results.csv"
        results_df.to_csv(output_csv_path, index=False)
        print(f"Saved detailed per-image RMSE results to '{output_csv_path}'")
        
        print("\nFirst 10 per-image RMSE results:")
        print(results_df.head(10).to_string(index=False))
    else:
        print("No results were generated.")