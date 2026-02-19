import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kuramoto", action="store_true", default=True)
    parser.add_argument("--kuramoto_path", type=Path, default=Path("kuramoto/data"))
    args = parser.parse_args()

    root = Path("kuramoto")
    root.mkdir(exist_ok=True)

    if args.kuramoto:
        print("Building Kuramoto dataset")
        out_dir = root / "data"
        out_dir.mkdir(exist_ok=True)

        for split in ["train", "valid", "test"]:
            print(f"Processing {split}...")

            # Load raw data
            labels_path = args.kuramoto_path / f"edges_{split}_kuramoto5.npy"
            data_path = args.kuramoto_path / f"feat_{split}_kuramoto5.npy"

            labels = np.load(labels_path)
            data = np.load(data_path)

            # Preprocess
            labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(3)
            data = torch.tensor(data[:, :, :, 1], dtype=torch.float32).swapaxes(1, 2)

            dataset = [[d, l] for d, l in zip(data, labels)]

            # Save
            save_name = "val.pt" if split == "valid" else f"{split}.pt"
            torch.save(dataset, out_dir / save_name)

            del dataset
    else:
        raise ValueError("Dataset not supported.")


if __name__ == "__main__":
    main()
