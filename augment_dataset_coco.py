import json
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import logging
import shutil
import random
from typing import Dict, List, Tuple
import argparse
import matplotlib.pyplot as plt
import os
import tempfile

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def preprocess_image(image: np.ndarray, invert_binary: bool = False) -> np.ndarray:
    """Preprocess image with adaptive thresholding and bordering
    
    Args:
        image: Input image array
        invert_binary: Whether to invert the binary threshold
        
    Returns:
        Preprocessed image array
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Apply adaptive thresholding
    threshold_type = cv2.THRESH_BINARY_INV if invert_binary else cv2.THRESH_BINARY
    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=threshold_type,
        blockSize=11,
        C=2
    )
    
    # Add border
    h, w = binary.shape
    bordered = np.zeros((h + 2, w + 2), dtype=np.uint8)
    bordered[1:-1, 1:-1] = binary
    
    # Convert back to RGB
    bordered_rgb = cv2.cvtColor(bordered, cv2.COLOR_GRAY2RGB)
    
    return bordered_rgb

def create_augmentation_pipeline() -> A.Compose:
    """Create an augmentation pipeline with various transformations"""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 7)),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.MultiplicativeNoise(),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
            A.ImageCompression(quality_lower=60, quality_upper=100),
            A.ToGray(p=0.2),
            A.ToSepia(p=0.2),
        ], p=0.5),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.2,
            p=0.5
        ),
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['category_ids']
    ))

def augment_dataset(
    dataset_dir: str,
    output_dir: str,
    augmentations_per_image: int = 3
) -> None:
    """
    Augment COCO dataset with various transformations
    
    Args:
        dataset_dir: Path to original dataset directory
        output_dir: Path to output directory for augmented dataset
        augmentations_per_image: Number of augmentations to create per image
    """
    setup_logging()
    
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    
    # Load original annotations
    with open(dataset_path / 'result.json', 'r') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    images_output_dir = output_path / 'images'
    images_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization directory
    viz_output_dir = output_path / 'visualizations'
    viz_output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_bbox(image: np.ndarray, bboxes: List[List[float]], category_ids: List[int], 
                      categories: List[Dict], filename: str) -> None:
        """Visualize all bounding boxes on image and save to visualization directory"""
        img_copy = image.copy()
        
        # Draw all bounding boxes
        for bbox, category_id in zip(bboxes, category_ids):
            x, y, w, h = map(int, bbox)
            
            # Get category name
            category_name = next(cat['name'] for cat in categories if cat['id'] == category_id)
            
            # Generate random color for this category
            color = tuple(random.randint(0, 255) for _ in range(3))
            
            # Draw rectangle
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
            
            # Put text with background for better visibility
            text_size = cv2.getTextSize(category_name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.rectangle(img_copy, (x, y - 30), (x + text_size[0], y), color, -1)
            cv2.putText(img_copy, category_name, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Save visualization
        viz_path = viz_output_dir / f"viz_{filename}"
        cv2.imwrite(str(viz_path), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
    
    # Initialize augmented dataset
    augmented_images = []
    augmented_annotations = []
    next_image_id = max(img['id'] for img in images) + 1
    next_ann_id = max(ann['id'] for ann in annotations) + 1
    
    # Create augmentation pipeline
    transform = create_augmentation_pipeline()
    
    # Log initial category distribution
    category_distribution = {}
    for ann in annotations:
        cat_id = ann['category_id']
        if cat_id in category_distribution:
            category_distribution[cat_id] += 1
        else:
            category_distribution[cat_id] = 1
    
    logging.info("Initial category distribution:")
    for cat_id, count in category_distribution.items():
        cat_name = next(cat['name'] for cat in categories if cat['id'] == cat_id)
        logging.info(f"Category {cat_id} ({cat_name}): {count} instances")

    # Process each image
    for img in images:
        image_path = dataset_path / 'images' / img['file_name']
        if not image_path.exists():
            logging.warning(f"Image not found: {image_path}")
            continue
            
        # Read image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        img_annotations = [ann for ann in annotations if ann['image_id'] == img['id']]
        
        # Create augmentations
        for aug_idx in range(augmentations_per_image):
            # Prepare bboxes and category ids for albumentations
            bboxes = [ann['bbox'] for ann in img_annotations]
            category_ids = [ann['category_id'] for ann in img_annotations]
            
            # Apply preprocessing first
            preprocessed = preprocess_image(image, invert_binary=random.random() > 0.5)
            
            # Apply augmentation
            transformed = transform(
                image=preprocessed,
                bboxes=bboxes,
                category_ids=category_ids
            )
            
            # Save augmented image
            aug_image_name = f"{Path(img['file_name']).stem}_aug_{aug_idx}{Path(img['file_name']).suffix}"
            cv2.imwrite(
                str(images_output_dir / aug_image_name),
                cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
            )
            
            # Create visualization for all bboxes in augmented image
            visualize_bbox(transformed['image'], transformed['bboxes'], 
                         transformed['category_ids'], categories, aug_image_name)
            
            # Create new image entry
            new_image = {
                'id': next_image_id,
                'file_name': aug_image_name,
                'height': transformed['image'].shape[0],
                'width': transformed['image'].shape[1]
            }
            augmented_images.append(new_image)
            
            # Create new annotations
            for bbox, category_id, orig_ann in zip(
                transformed['bboxes'],
                transformed['category_ids'],
                img_annotations
            ):
                new_ann = orig_ann.copy()
                new_ann['id'] = next_ann_id
                new_ann['image_id'] = next_image_id
                new_ann['bbox'] = list(map(float, bbox))
                augmented_annotations.append(new_ann)
                next_ann_id += 1
                
            next_image_id += 1
            
        # Copy original image
        shutil.copy2(image_path, images_output_dir / img['file_name'])
        
        # Create visualization for original image
        original_bboxes = [ann['bbox'] for ann in img_annotations]
        original_category_ids = [ann['category_id'] for ann in img_annotations]
        visualize_bbox(image, original_bboxes, original_category_ids, categories, img['file_name'])
    
    # Combine original and augmented data
    output_data = {
        'images': images + augmented_images,
        'annotations': annotations + augmented_annotations,
        'categories': categories
    }
    
    # Save augmented annotations
    with open(output_path / 'result.json', 'w') as f:
        json.dump(output_data, f, indent=4)
    
    # Calculate final category distribution
    final_category_distribution = {}
    for ann in (annotations + augmented_annotations):
        cat_id = ann['category_id']
        if cat_id in final_category_distribution:
            final_category_distribution[cat_id] += 1
        else:
            final_category_distribution[cat_id] = 1

    logging.info(f"Augmentation completed:")
    logging.info(f"Categories found: {len(categories)}")
    for cat in categories:
        initial_count = category_distribution.get(cat['id'], 0)
        final_count = final_category_distribution.get(cat['id'], 0)
        augmented_count = final_count - initial_count
        logging.info(f"Category {cat['id']} ({cat['name']}): "
                    f"Initial: {initial_count}, "
                    f"Augmented: {augmented_count}, "
                    f"Final: {final_count}")
    logging.info(f"Original images: {len(images)}")
    logging.info(f"Augmented images created: {len(augmented_images)}")
    logging.info(f"Total images: {len(images) + len(augmented_images)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment COCO dataset")
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
        help="Path to the output directory where augmented dataset will be saved"
    )
    parser.add_argument(
        '--augmentations_per_image',
        type=int,
        default=3,
        help="Number of augmentations to create per image (default: 3)"
    )
    
    args = parser.parse_args()
    
    try:
        augment_dataset(
            args.dataset_dir,
            args.output_dir,
            args.augmentations_per_image
        )
    except Exception as e:
        logging.error(f"Error: {e}")
        raise

#EXAMPLE : python augment_dataset_coco.py --dataset_dir ./coco-dataset --output_dir ./augmented-dataset --augmentations_per_image 5
