import os
import argparse
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src import model_utils
from src.dataset import AgricultureVisionDataset, KomatsunaDataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default="komatsuna-DatasetNinja")
    parser.add_argument("--dataset_type", type=str, default="komatsuna", choices=["agriculture_vision", "komatsuna"])
    parser.add_argument("--weight_folder", type=str, default="./results")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./results/Fold_1")
    parser.add_argument("--num_samples", type=int, default=3)
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cpu") # run visualization on CPU for reliability
    
    # Load config
    conf_path = os.path.join(args.weight_folder, "conf.json")
    if not os.path.exists(conf_path):
        print(f"Error: configuration file not found at {conf_path}")
        return
        
    with open(conf_path, "r") as f:
        model_config = json.loads(f.read())
        
    config = argparse.Namespace(**model_config)
    config.input_dim = 3
    config.device = "cpu"
    
    # Load model
    print("Loading model...")
    model = model_utils.get_model(config, mode="semantic")
    weight_path = os.path.join(args.weight_folder, f"Fold_{args.fold}", "model.pth.tar")
    if not os.path.exists(weight_path):
        print(f"Error: weights not found at {weight_path}")
        return
        
    sd = torch.load(weight_path, map_location=device)
    model.load_state_dict(sd["state_dict"])
    model.eval()
    
    # Load dataset
    if args.dataset_type == "komatsuna":
        dataset = KomatsunaDataset(
            folder=args.dataset_folder,
            norm=True,
            folds=[args.fold]
        )
        raw_dataset = KomatsunaDataset(
            folder=args.dataset_folder,
            norm=False,
            folds=[args.fold]
        )
        # Custom colormap for Komatsuna classes: 0: background (black), 1: leaf (green), 2: stem (blue)
        cmap = ListedColormap(['black', 'green', 'blue'])
        vmin, vmax = 0, 2
    else:
        dataset = AgricultureVisionDataset(
            folder=args.dataset_folder,
            norm=True,
            folds=[args.fold]
        )
        raw_dataset = AgricultureVisionDataset(
            folder=args.dataset_folder,
            norm=False,
            folds=[args.fold]
        )
        cmap = "tab20"
        vmin, vmax = 0, 19
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize samples
    num_samples = min(args.num_samples, len(dataset))
    print(f"Visualizing {num_samples} samples...")
    
    for i in range(num_samples):
        # Get normalized input for model
        (x, dates), _ = dataset[i]
        
        # Get raw input for plotting
        (raw_x, _), _ = raw_dataset[i]
        
        # Run inference
        with torch.no_grad():
            out = model(x.unsqueeze(0), batch_positions=dates.unsqueeze(0))
            pred = out.argmax(dim=1).squeeze(0).numpy() # shape (256, 256)
            
        # Re-format raw image to numpy RGB [0, 1]
        raw_img = raw_x.squeeze(0).permute(1, 2, 0).numpy()
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Original Image
        axes[0].imshow(raw_img)
        axes[0].set_title(f"Original Image (Sample {i+1})")
        axes[0].axis("off")
        
        # Predicted Mask
        im = axes[1].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title("Predicted Segmentation Mask")
        axes[1].axis("off")
        
        # Colorbar
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        output_path = os.path.join(args.output_dir, f"prediction_sample_{i+1}.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    main()
