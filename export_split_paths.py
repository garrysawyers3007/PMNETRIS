import json
import numpy as np
import random
import torch
from torch.utils.data import random_split
from dataloader import PMnet_data_usc
from config import config_USC_RISMapNet_V1


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    data_root = "datasetRIS_32x32/"
    set_seed(1234)

    cfg = config_USC_RISMapNet_V1()

    ris_pos_min = [-400, -400, 0]
    ris_pos_max = [400, 400, 55]
    data_usc_train = PMnet_data_usc(
        dir_dataset=data_root,
        ris_pos_min=ris_pos_min,
        ris_pos_max=ris_pos_max,
    )

    dataset_size = len(data_usc_train)
    train_size = int(dataset_size * cfg.train_ratio)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(data_usc_train, [train_size, test_size])

    # noris_power_map_path from train indices
    train_noris_paths = [
        data_usc_train.metadata_records[i]["noris_power_map_path"]
        for i in train_dataset.indices
    ]

    # power_map_path from test indices (only "noRIS" type)
    test_power_paths = [
        data_usc_train.metadata_records[i]["power_map_path"]
        for i in test_dataset.indices
        if data_usc_train.metadata_records[i]["type"] == "noRIS"
    ]

    # Common paths (appear in both sets)
    common_paths = set(train_noris_paths) & set(test_power_paths)

    print(f"Train samples: {len(train_noris_paths)}")
    print(f"Test samples:  {len(test_power_paths)}")
    print(f"Common paths:  {len(common_paths)}")

    # Save to JSON
    output = {
        "train_noris_power_map_paths": train_noris_paths,
        "test_power_map_paths": test_power_paths,
        "common_paths": sorted(common_paths),
    }

    with open("split_paths.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Saved to split_paths.json")
