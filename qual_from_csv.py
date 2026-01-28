import os
import numpy as np
import torch
from torchvision import transforms
from skimage import io
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from network.pmnet import PMNetFiLM
import json
import matplotlib.patches as patches

def generate_predictions_from_csv(model, csv_path, output_dir, ris_pos_min, ris_pos_max, device, rx_box_size=10):
    """
    Reads a CSV file, runs inference for each entry, and saves the output images.
    """
    model.eval()
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True)
    ])
    
    img_size_final = 256
    ris_pos_min = np.array(ris_pos_min, dtype=np.float32)
    ris_pos_max = np.array(ris_pos_max, dtype=np.float32)

    pbar = tqdm(df.iterrows(), total=len(df), desc="Generating Predictions")

    with torch.no_grad():
        for index, row in pbar:
            try:
                city_map = np.asarray(io.imread(row['city_map_path']))
                if city_map.ndim == 3: city_map = city_map[:,:,0]
                
                tx_map = np.asarray(io.imread(row['tx_map_path']))
                if tx_map.ndim == 3: tx_map = tx_map[:,:,0]

                # Load the ground truth power map for the difference calculation
                power_map_gt = np.asarray(io.imread(row['power_map_path']))
                if power_map_gt.ndim == 3: power_map_gt = power_map_gt[:,:,0]

            except FileNotFoundError as e:
                print(f"\nWarning: Could not find image file {e.filename} for row {index}. Skipping.")
                continue

            input_np = np.stack([city_map, tx_map], axis=-1)
            inputs_tensor = transform(input_np).type(torch.float32).unsqueeze(0).to(device)
            power_gt_tensor = transform(power_map_gt).type(torch.float32).unsqueeze(0).to(device)

            ris_pos_list = json.loads(row['ris_true_world_pos'])
            ris_orientation_list = json.loads(row['ris_orientation_rpy'])
            rx_pos_list = json.loads(row['rx_true_world_pos'])

            ris_pos_norm_x, ris_pos_norm_y, ris_pos_norm_z = ris_pos_list
            ris_roll, ris_pitch, ris_yaw = ris_orientation_list
            rx_pos_norm_x, rx_pos_norm_y, rx_pos_norm_z = rx_pos_list
            ris_flag = row['is_ris_present']

            cm_dim_original = 900.0  # Original city map dimension in pixels
            ris_pos_range = ris_pos_max - ris_pos_min

            bias_orientation = {0: [50,50], 1: [50,-50], 2: [-50,50], 3: [-50,-50]}
            orientation = int(row['city_map_path'].split('_')[-1].split('.')[0])

            sim_ris_pos_x = (ris_pos_norm_x * ris_pos_range[0]) + ris_pos_min[0] + bias_orientation[orientation][0]
            sim_ris_pos_y = (ris_pos_norm_y * ris_pos_range[1]) + ris_pos_min[1] + bias_orientation[orientation][1]
            sim_rx_pos_x = (rx_pos_norm_x * ris_pos_range[0]) + ris_pos_min[0] + bias_orientation[orientation][0]
            sim_rx_pos_y = (rx_pos_norm_y * ris_pos_range[1]) + ris_pos_min[1] + bias_orientation[orientation][1]

            # 2. Map sim coords to original 900x900 pixel grid
            ris_pixel_x_900 = sim_ris_pos_x + cm_dim_original / 2.0
            ris_pixel_y_900 = cm_dim_original / 2.0 - sim_ris_pos_y # Y is inverted
            rx_pixel_x_900 = sim_rx_pos_x + cm_dim_original / 2.0
            rx_pixel_y_900 = cm_dim_original / 2.0 - sim_rx_pos_y # Y is inverted

            # 3. Scale to final 256x256 image
            scale_factor = img_size_final / cm_dim_original
            ris_pixel_x = int(round(ris_pixel_x_900 * scale_factor))
            ris_pixel_y = int(round(ris_pixel_y_900 * scale_factor))
            rx_pixel_x = int(round(rx_pixel_x_900 * scale_factor))
            rx_pixel_y = int(round(rx_pixel_y_900 * scale_factor))
            
            # The model expects normalized coordinates, which are taken directly from the CSV
            ris_params_list = [ris_pos_norm_x, -ris_pos_norm_y, ris_pos_norm_z, ris_roll, ris_pitch, ris_yaw, rx_pos_norm_x, rx_pos_norm_y, rx_pos_norm_z, ris_flag]
            ris_params_tensor = torch.tensor(ris_params_list, dtype=torch.float32).unsqueeze(0).to(device)

            pred = model(inputs_tensor, ris_params_tensor)
            pred = torch.clip(pred, 0, 1)

            # --- Calculate RMSE for this specific prediction ---
            # rmse_loss = torch.sqrt(torch.mean((pred - power_gt_tensor) ** 2))
            rmse_value = float(row['rmse'])

            # --- Visualize and Save Output ---
            plt.figure(figsize=(18, 6))

            tx_map_for_display = inputs_tensor[0, 1].cpu().numpy() * 255
            tx_pred_y, tx_pred_x = np.unravel_index(np.argmax(tx_map_for_display), tx_map_for_display.shape)


           # Plot 1: Ground Truth Power Map
            ax1 = plt.subplot(1, 2, 1)
            plt.axis("off")
            gt_display = (power_gt_tensor[0, 0].cpu().numpy() * 255 + tx_map_for_display)
            ax1.imshow(gt_display, cmap='gray', vmin=0, vmax=255)
            ax1.set_title("Ground Truth Power Map")
            ax1.scatter(tx_pred_x, tx_pred_y, c='green', s=120, marker='^', label='TX', edgecolor='black', linewidth=0.8)
            # --- ADDED MARKERS ---
            if ris_flag == 1.0:
                ax1.scatter(ris_pixel_x, ris_pixel_y, c='gold', s=300, marker='*', label='RIS', edgecolor='black', linewidth=0.8, alpha=0.9)
                ax1.scatter(rx_pixel_x, rx_pixel_y, c='blue', s=100, marker='o', label='RX', edgecolor='black', linewidth=0.8)

            # Plot 2: Predicted Power Map
            ax2 = plt.subplot(1, 2, 2)
            plt.axis("off")
            pred_display = (pred[0, 0].cpu().numpy() * 255 + tx_map_for_display)
            ax2.imshow(pred_display, cmap='gray', vmin=0, vmax=255)
            ax2.scatter(tx_pred_x, tx_pred_y, c='green', s=120, marker='^', label='TX', edgecolor='black', linewidth=0.8)

            title_ris_info = "NoRIS "
            if ris_flag == 1.0:
                # De-normalize RIS coords ONLY for the title text
                norm_coords = np.array([ris_pos_norm_x, ris_pos_norm_y, ris_pos_norm_z], dtype=np.float32)
                range_val = ris_pos_max - ris_pos_min
                denorm_coords = (norm_coords * range_val) + ris_pos_min
                title_ris_info = f"RIS@[{denorm_coords[0]:.1f}, {denorm_coords[1]:.1f}, {denorm_coords[2]:.1f}] "
                # --- ADDED MARKERS ---
                ax2.scatter(ris_pixel_x, ris_pixel_y, c='gold', s=300, marker='*', label='RIS', edgecolor='black', linewidth=0.8, alpha=0.9)
                ax2.scatter(rx_pixel_x, rx_pixel_y, c='blue', s=100, marker='o', edgecolor='black', linewidth=0.8)

            ax2.set_title(f"Predicted\n(RMSE: {rmse_value:.4f})")

            # --- Save Figure ---
            plt.tight_layout() # Added for better spacing
            output_filename = os.path.join(output_dir, f"prediction_{index}.png")
            plt.savefig(output_filename)
            plt.close()

    print(f"\nInference complete. All prediction images saved to '{output_dir}'.")


if __name__ == '__main__':
    CSV_INPUT_PATH = "test_rmse_results.csv"
    MODEL_PATH = "datasetRISNewCorrected/PMNet_results/augmented_config_USC_pmnetV3_V2_epoch100/8_0.0001_0.45_10/model_0.03808.pt"
    OUTPUT_IMAGE_DIR = "inference_predictions_new"

    RIS_POS_MIN = [-400.0, -400.0, 0.0] 
    RIS_POS_MAX = [400.0, 400.0, 55.0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PMNetFiLM(
        n_blocks=[3, 3, 27, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=8,
        cond_features=10).to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model weights not found at '{MODEL_PATH}'. Using a new dummy model.")
    else:
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"Successfully loaded model weights from '{MODEL_PATH}'.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            exit()

    generate_predictions_from_csv(
        model=model,
        csv_path=CSV_INPUT_PATH,
        output_dir=OUTPUT_IMAGE_DIR,
        ris_pos_min=RIS_POS_MIN,
        ris_pos_max=RIS_POS_MAX,
        device=device
    )