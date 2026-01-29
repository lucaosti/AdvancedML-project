"""Prepare and organize LineMOD dataset."""

import os
import argparse
import random
import glob


def prepare_linemod_data(source_dir, output_dir, train_ratio=0.8, seed=42):
    """Organize LineMOD dataset and generate train/test splits."""
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    object_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    
    print(f"Found {len(object_folders)} objects in {source_dir}")
    print(f"Splitting data: {train_ratio*100:.0f}% Train / {(1-train_ratio)*100:.0f}% Validation")

    for obj_name in object_folders:
        obj_path = os.path.join(source_dir, obj_name)
        
        rgb_path = os.path.join(obj_path, 'rgb')
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(obj_path, 'JPEGImages')
            if not os.path.exists(rgb_path):
                print(f"Skipping {obj_name}: No 'rgb' or 'JPEGImages' folder found.")
                continue

        files = glob.glob(os.path.join(rgb_path, "*.*"))
        valid_exts = ['.png', '.jpg', '.jpeg']
        ids = [os.path.splitext(os.path.basename(f))[0] for f in files if os.path.splitext(f)[1].lower() in valid_exts]
        ids.sort()

        if not ids:
            print(f"Skipping {obj_name}: No images found.")
            continue

        random.seed(seed)
        random.shuffle(ids)

        split_idx = int(len(ids) * train_ratio)
        train_ids = ids[:split_idx]
        test_ids = ids[split_idx:]

        target_obj_dir = os.path.join(output_dir, obj_name)
        os.makedirs(target_obj_dir, exist_ok=True)

        with open(os.path.join(target_obj_dir, 'train.txt'), 'w') as f:
            f.write('\n'.join(train_ids))
        
        with open(os.path.join(target_obj_dir, 'test.txt'), 'w') as f:
            f.write('\n'.join(test_ids))

        print(f"[{obj_name}] Total: {len(ids)} | Train: {len(train_ids)} | Val: {len(test_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LineMOD dataset")
    parser.add_argument("--source", type=str, required=True, help="Source dataset path")
    parser.add_argument("--output", type=str, default="data/", help="Output directory")
    parser.add_argument("--ratio", type=float, default=0.8, help="Training ratio (default: 0.8)")
    
    args = parser.parse_args()
    prepare_linemod_data(args.source, args.output, args.ratio)
