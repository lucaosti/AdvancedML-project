"""Fine-tune YOLO model on LineMOD dataset for 13-class object detection."""

import os
import sys
import shutil
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from rgb_pipeline.config import YOLO_NAME_TO_LINEMOD_ID, OBJECT_ID_TO_NAME, YOLO_FINETUNED_ID_MAP


def get_bbox(mask_path):
    """Extract normalized bounding box from mask image in YOLO format."""
    mask = np.array(Image.open(mask_path).convert('L'))

    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return None

    y1, x1 = np.min(rows), np.min(cols)
    y2, x2 = np.max(rows), np.max(cols)

    h, w = mask.shape

    box_w = x2 - x1
    box_h = y2 - y1
    box_cx = x1 + (box_w / 2)
    box_cy = y1 + (box_h / 2)

    return [box_cx/w, box_cy/h, box_w/w, box_h/h]


def prepare_yolo_data(data_root, output_dir, linemod_objects):
    """Convert LineMOD dataset to YOLO training format."""
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    
    if output_dir.exists():
        print(f"Cleaning existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    out_path = output_dir
    
    yolo_id_to_linemod_id = {}
    for yolo_id, class_name in YOLO_FINETUNED_ID_MAP.items():
        linemod_id = YOLO_NAME_TO_LINEMOD_ID[class_name]
        yolo_id_to_linemod_id[yolo_id] = linemod_id
    
    for yolo_class_idx in sorted(yolo_id_to_linemod_id.keys()):
        obj_id = yolo_id_to_linemod_id[yolo_class_idx]
        name = linemod_objects[obj_id]
        base = data_root / obj_id
        
        if not base.exists():
            print(f"Warning: Object directory {base} does not exist, skipping...")
            continue

        class_idx = yolo_class_idx

        for split in ['train', 'test']:
            (out_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (out_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

            txt_path = base / f"{split}.txt"
            if not txt_path.exists():
                print(f"Warning: {txt_path} does not exist, skipping {name} {split} split...")
                continue

            with open(txt_path) as f:
                ids = [x.strip() for x in f.readlines()]

            for idx in tqdm(ids, desc=f"{name} {split}"):
                src_img = base / 'rgb' / f"{int(idx):04d}.png"
                src_mask = base / 'mask' / f"{int(idx):04d}.png"
                dst_img = out_path / 'images' / split / f"{name}_{idx}.png"
                dst_txt = out_path / 'labels' / split / f"{name}_{idx}.txt"

                if src_img.exists():
                    shutil.copy(src_img, dst_img)
                else:
                    print(f"Warning: Image {src_img} does not exist")
                    continue
                    
                if src_mask.exists():
                    bbox = get_bbox(src_mask)
                    if bbox:
                        with open(dst_txt, 'w') as f:
                            f.write(f"{class_idx} {' '.join(map(str, bbox))}\n")
                else:
                    print(f"Warning: Mask {src_mask} does not exist, skipping label...")

    class_names = [YOLO_FINETUNED_ID_MAP[i] for i in sorted(YOLO_FINETUNED_ID_MAP.keys())]
    yaml_config = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/test',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    
    print(f"\nYOLO dataset prepared successfully!")
    print(f"Dataset YAML: {yaml_path}")
    print(f"Classes: {class_names}")
    print(f"Total classes: {len(class_names)}")


def train_yolo(
    data_yaml,
    epochs=50,
    imgsz=640,
    batch=16,
    model='yolov8n.pt',
    project_dir=None,
    name='linemod_yolo',
    resume=False,
    resume_from=None
):
    """Train YOLO model on LineMOD dataset."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found. Install it with: pip install ultralytics")
        sys.exit(1)
    
    if project_dir is None:
        project_dir = os.path.join(project_root, 'yolo', 'weights')
    
    if resume:
        if resume_from and os.path.exists(resume_from):
            checkpoint_path = resume_from
        else:
            import glob
            search_pattern = os.path.join(project_dir, '**', 'last.pt')
            found_files = glob.glob(search_pattern, recursive=True)
            
            if not found_files:
                print("Error: No checkpoint found for resuming. Starting new training...")
                resume = False
            else:
                found_files.sort(key=os.path.getmtime, reverse=True)
                checkpoint_path = found_files[0]
                print(f"Resuming from checkpoint: {checkpoint_path}")
        
        if resume:
            model = YOLO(checkpoint_path)
            model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                project=project_dir,
                name=name,
                resume=True
            )
            return
    
    print(f"Starting new training with model: {model}")
    yolo_model = YOLO(model)
    yolo_model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project_dir,
        name=name
    )
    
    print("\nTraining finished!")


def copy_best_weights(run_dir, output_path):
    """Copy best weights from training run to output path."""
    run_dir = Path(run_dir)
    output_path = Path(output_path)
    
    weights_dir = run_dir / 'weights'
    best_weights = weights_dir / 'best.pt'
    
    if not best_weights.exists():
        print(f"Warning: best.pt not found at {best_weights}")
        last_weights = weights_dir / 'last.pt'
        if last_weights.exists():
            print(f"Using last.pt instead: {last_weights}")
            best_weights = last_weights
        else:
            print(f"Error: No weights found in {weights_dir}")
            return False
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.copy(best_weights, output_path)
    print(f"Copied best weights to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO model on LineMOD dataset"
    )
    
    parser.add_argument('--data_root', type=str, 
                       default='data/Linemod_preprocessed/data',
                       help='Path to Linemod_preprocessed/data/ directory')
    parser.add_argument('--output_dir', type=str, 
                       default='yolo_data',
                       help='Output directory for YOLO dataset')
    parser.add_argument('--prepare_only', action='store_true',
                       help='Only prepare data, do not train')
    
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Base YOLO model (yolov8n.pt, yolov8s.pt, etc.)')
    parser.add_argument('--project_dir', type=str, default=None,
                       help='Directory for saving training outputs (default: yolo/runs)')
    parser.add_argument('--name', type=str, default='linemod_yolo',
                       help='Name for this training run')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from latest checkpoint')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to specific checkpoint to resume from')
    
    parser.add_argument('--copy_weights', action='store_true', default=True,
                       help='Copy best weights to yolo/weights/fine-tuned-yolo-weights.pt')
    parser.add_argument('--weights_output', type=str, 
                       default='yolo/weights/fine-tuned-yolo-weights.pt',
                       help='Output path for final weights')
    
    args = parser.parse_args()
    
    linemod_objects = {}
    for yolo_id, class_name in YOLO_FINETUNED_ID_MAP.items():
        linemod_id = YOLO_NAME_TO_LINEMOD_ID[class_name]
        linemod_objects[linemod_id] = class_name
    
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: Data root directory does not exist: {data_root}")
        print("Please provide the correct path to Linemod_preprocessed/data/")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    data_yaml = output_dir / 'dataset.yaml'
    
    if data_yaml.exists() and not args.prepare_only:
        print(f"YOLO dataset already exists at {output_dir}")
        print("Skipping data preparation. Use --prepare_only to force re-preparation.")
    else:
        print("Preparing YOLO training data...")
        prepare_yolo_data(data_root, output_dir, linemod_objects)
    
    if args.prepare_only:
        print("Data preparation complete. Exiting.")
        return
    
    if not data_yaml.exists():
        print(f"Error: Dataset YAML not found at {data_yaml}")
        print("Please run data preparation first.")
        sys.exit(1)
    
    print("\nStarting YOLO training...")
    train_yolo(
        data_yaml=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model=args.model,
        project_dir=args.project_dir,
        name=args.name,
        resume=args.resume,
        resume_from=args.resume_from
    )
    
    if args.copy_weights:
        if args.project_dir:
            run_dir = Path(args.project_dir) / args.name
        else:
            run_dir = Path(project_root) / 'yolo' / 'weights' / args.name
        
        copy_best_weights(run_dir, args.weights_output)


if __name__ == "__main__":
    main()
