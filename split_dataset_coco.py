import json
import os
import random
from shutil import copyfile
import argparse
import logging
from typing import Dict, List, Set
from pathlib import Path

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def validate_split_ratio(split_ratio: float) -> None:
    if not 0 < split_ratio < 1:
        raise ValueError("Split ratio must be between 0 and 1")

def split_dataset(dataset_dir: str, output_dir: str, split_ratio: float = 0.8) -> None:
    setup_logging()
    validate_split_ratio(split_ratio)
    
    # Convert to Path objects for better path handling
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    result_json_path = dataset_path / 'result.json'

    if not result_json_path.exists():
        raise FileNotFoundError(f"result.json not found in {dataset_dir}")

    try:
        with open(result_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except IOError as e:
        logging.error(f"Failed to read result.json: {e}")
        raise

    try:
        images = data['images']
        annotations = data['annotations']
        categories = data['categories']
    except KeyError as e:
        raise KeyError(f"Missing required key in result.json: {e}")

    if not images:
        raise ValueError("No images found in the dataset")

    # Shuffle data with a fixed seed for reproducibility
    random.seed(42)
    random.shuffle(images)
    
    train_size = int(len(images) * split_ratio)
    train_images = images[:train_size]
    val_images = images[train_size:]
    
    def filter_annotations(images_subset: List[Dict]) -> List[Dict]:
        image_ids: Set[int] = {img['id'] for img in images_subset}
        return [ann for ann in annotations if ann['image_id'] in image_ids]
    
    # Prepare output data
    train_data = {
        "images": train_images,
        "annotations": filter_annotations(train_images),
        "categories": categories
    }
    
    val_data = {
        "images": val_images,
        "annotations": filter_annotations(val_images),
        "categories": categories
    }
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save JSON files
    try:
        with open(output_path / 'train.json', 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=4)
        
        with open(output_path / 'val.json', 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=4)
    except IOError as e:
        logging.error(f"Failed to write JSON files: {e}")
        raise
    
    # Create image directories
    train_images_dir = output_path / 'train' / 'images'
    val_images_dir = output_path / 'val' / 'images'
    
    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    
    def copy_images(images: List[Dict], target_dir: Path, source_dir: Path) -> int:
        missing_count = 0
        for img in images:
            # Remove 'images/' prefix from file_name if present
            file_name = Path(img['file_name']).name
            source_path = source_dir / 'images' / file_name
            target_path = target_dir / file_name
            
            if source_path.exists():
                try:
                    copyfile(source_path, target_path)
                except IOError as e:
                    logging.error(f"Failed to copy {source_path}: {e}")
                    missing_count += 1
            else:
                logging.warning(f"Source image not found: {source_path}")
                missing_count += 1
        return missing_count

    missing_train = copy_images(train_images, train_images_dir, dataset_path)
    missing_val = copy_images(val_images, val_images_dir, dataset_path)

    logging.info(f"Dataset split completed:")
    logging.info(f"Training data: {len(train_images)} images ({missing_train} missing)")
    logging.info(f"Validation data: {len(val_images)} images ({missing_val} missing)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split COCO dataset into training and validation sets")
    
    parser.add_argument(
        '--dataset_dir', 
        required=True, 
        type=str,
        help="Path to the dataset directory containing images and result.json"
    )
    parser.add_argument(
        '--output_dir', 
        required=True,
        type=str,
        help="Path to the output directory where split dataset will be saved"
    )
    parser.add_argument(
        '--split_ratio', 
        type=float, 
        default=0.8,
        help="Ratio of training set (default: 0.8)"
    )
    
    args = parser.parse_args()
    
    try:
        split_dataset(args.dataset_dir, args.output_dir, args.split_ratio)
    except Exception as e:
        logging.error(f"Error: {e}")
        raise

# EXAMPLE : python split_dataset_coco.py --dataset_dir /path/to/dataset --output_dir /path/to/output --split_ratio 0.8
