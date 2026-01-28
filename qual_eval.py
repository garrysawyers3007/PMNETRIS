import json
import os
import numpy as np
import random
import torch
from torchvision import transforms
from skimage import io
from network.pmnet import PMNet
import matplotlib.pyplot as plt

json_path = '/mnt/c/Users/gaura/Documents/pmnet-sionna-rt/datasetRISNew/metadata.json'
with open(json_path, 'r') as f:
    data = json.load(f)

processed_data_0 = [point for point in data if point['tx_id']== '34']
processed_data_20 = [point for point in data if point['tx_id']== '5']

data_no_ris_0 = processed_data_0[0]
data_ris_0 = random.choice(processed_data_0[1:])

data_no_ris_20 = processed_data_20[0]
data_ris_20 = random.choice(processed_data_20[1:])

final_data = [data_no_ris_0, data_ris_0, data_no_ris_20, data_ris_20]

ris_pos_min=np.array([-400, -400, 0], dtype=np.float32)
ris_pos_max=np.array([400, 400, 55], dtype=np.float32)

def normalize_ris_pos(ris_pos_array):
    range_val = ris_pos_max - ris_pos_min
    normalized_pos = (ris_pos_array - ris_pos_min) / range_val
    return normalized_pos

image_transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Resize((256, 256), antialias=True) 
                 ])

load_model = 'datasetRISNew/PMNet_results/augmented_config_USC_pmnetV3_V2_epoch100/8_0.0001_0.45_10/model_0.03856.pt'
model = PMNet(
    n_blocks=[3, 3, 27, 3],
    atrous_rates=[6, 12, 18],
    multi_grids=[1, 2, 4],
    output_stride=8,
    cond_features=7)
model.cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(load_model))
model.to(device)

model.eval()

points = []
for data_point in final_data:
    image_buildings = np.asarray(io.imread(data_point['paths']["city_map"][0])) 
    if image_buildings.ndim == 3 and image_buildings.shape[-1] >= 3:
        image_buildings = image_buildings[:,:,0]

    image_tx = np.asarray(io.imread(data_point['paths']["tx_map"][0])) 
    if image_tx.ndim == 3 and image_tx.shape[-1] >= 3:
        image_tx = image_tx[:,:,0]

    image_power = np.asarray(io.imread(data_point['paths']["power_map"][0]))
    if image_power.ndim == 3 and image_power.shape[-1] >= 3:
        image_power = image_power[:,:,0]

    inputs_np = np.stack([image_buildings, image_tx], axis=-1)

    print(data_point["ris_true_world_pos"])
    data_point["ris_true_world_pos"] = np.array(data_point["ris_true_world_pos"], dtype=np.float32) if data_point["ris_true_world_pos"] else np.zeros(3, dtype=np.float32)
    data_point["ris_true_world_orientation_rpy"] = np.array(data_point["ris_true_world_orientation_rpy"], dtype=np.float32) if data_point["ris_true_world_orientation_rpy"] else np.zeros(3, dtype=np.float32)
    ris_info_list = list(normalize_ris_pos(data_point["ris_true_world_pos"])) + \
                        list([-x for x in data_point["ris_true_world_orientation_rpy"]]) + \
                        [1.0 if data_point["type"]=="RIS" else 0.0]
    
    ris_info_tensor = torch.tensor(ris_info_list, dtype=torch.float32)
    inputs_tensor = inputs_np 
    power_tensor_np = image_power

    inputs_tensor = image_transform(inputs_np).type(torch.float32)
            
    if power_tensor_np.dtype == np.uint8:
        power_tensor = image_transform(power_tensor_np).type(torch.float32)
    else: 
        power_tensor = image_transform(power_tensor_np.astype(np.uint8)).type(torch.float32)
    points.append((inputs_tensor, ris_info_tensor, power_tensor))

with torch.no_grad():
    for i, (inputs, ris_params, targets) in enumerate(points):
        inputs = inputs.unsqueeze(0).cuda()
        ris_params = ris_params.unsqueeze(0).cuda()
        targets = targets.unsqueeze(0).cuda()

        preds = model(inputs, ris_params)
        preds = torch.clip(preds, 0, 1)

        tx_map = inputs[0, 1].cpu().detach().numpy() * 255
        ris_params_np = ris_params[0].cpu().detach().numpy()
        ris_pos = ris_params_np[:3] * (np.array(ris_pos_max) - np.array(ris_pos_min)) + np.array(ris_pos_min) if ris_params_np[6] > 0 else [0, 0, 0]
        plt.figure(figsize=(15,10))
        plt.subplot(1,3,1)
        plt.axis("off")
        plt.title("Ground Truth" + f' (RIS pos: {ris_pos[0]:.1f}, {ris_pos[1]:.1f}, {ris_pos[2]:.1f})')
        left=(targets[0].cpu().detach().numpy()*255 + tx_map)[0][:,::-1]
        plt.imshow(left, cmap='gray', vmin=0, vmax=255)

        plt.subplot(1,3,2)
        plt.axis("off")
        plt.title("Predicted Image")
        right=(preds[0, 0].cpu().detach().numpy()*255 + tx_map)[:,::-1]
        plt.imshow(right, cmap='gray', vmin=0, vmax=255)

        img_name=os.path.join(f'eval_imgs_1/{i}.png')
        plt.savefig(img_name)