import json
import os
import shutil
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom
import cv2

def convert_coco_to_pascal(coco_annotations_path, coco_images_dir, pascal_voc_output_path):
    # Read COCO dataset annotations
    with open(coco_annotations_path, 'r') as annotations_file:
        coco_annotations = json.load(annotations_file)
    
    coco_images = coco_annotations['images']
    
    # Ensure the root output directory exists
    os.makedirs(pascal_voc_output_path, exist_ok=True)

    # Save class names in the root output directory
    categories = coco_annotations['categories']
    class_names_path = os.path.join(pascal_voc_output_path, "class_names.txt")
    with open(class_names_path, "w") as class_file:
        for category in categories:
            class_file.write(f"{category['id']}: {category['name']}\n")
    train_images_dir = os.path.join(pascal_voc_output_path, "train", "JPEGImages")
    train_annotations_dir = os.path.join(pascal_voc_output_path, "train", "Annotations")
    val_images_dir = os.path.join(pascal_voc_output_path, "val", "JPEGImages")
    val_annotations_dir = os.path.join(pascal_voc_output_path, "val", "Annotations")
    train_annotated_dir = os.path.join(pascal_voc_output_path, "train", "annotated")
    val_annotated_dir = os.path.join(pascal_voc_output_path, "val", "annotated")
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_annotated_dir, exist_ok=True)
    os.makedirs(train_annotations_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_annotated_dir, exist_ok=True)
    os.makedirs(val_annotations_dir, exist_ok=True)

    # Create a dictionary to map image IDs to file names
    image_id_to_filename = {image['id']: image['file_name'] for image in coco_images}

    # Shuffle images for random train/val split
    random.shuffle(coco_images)

    # Calculate split index
    split_index = int(len(coco_images) * 0.8)

    # Split images into train and validation sets
    train_images = coco_images[:split_index]
    val_images = coco_images[split_index:]

    # Create a dictionary to map image IDs to file names for train and val
    train_image_id_to_filename = {image['id']: image['file_name'] for image in train_images}
    val_image_id_to_filename = {image['id']: image['file_name'] for image in val_images}

    # Group annotations by image_id
    annotations_by_image = {}
    for annotation in coco_annotations['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)

    # Process each image
    for image in coco_images:
        image_id = image['id']
        image_filename = image['file_name']
        image_info = next(item for item in coco_images if item['id'] == image_id)

        # Create Pascal VOC XML structure
        annotation_xml = ET.Element("annotation")
        ET.SubElement(annotation_xml, "folder").text = "images"
        ET.SubElement(annotation_xml, "filename").text = image_filename

        size = ET.SubElement(annotation_xml, "size")
        ET.SubElement(size, "width").text = str(image_info['width'])
        ET.SubElement(size, "height").text = str(image_info['height'])
        ET.SubElement(size, "depth").text = "3"  # Assuming RGB images

        # Draw bounding boxes and add to XML
        image_path = os.path.join(coco_images_dir, image_filename)
        image = cv2.imread(image_path)

        for annotation in annotations_by_image.get(image_id, []):
            obj = ET.SubElement(annotation_xml, "object")
            ET.SubElement(obj, "name").text = str(annotation['category_id'])
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(annotation['bbox'][0]))
            ET.SubElement(bndbox, "ymin").text = str(int(annotation['bbox'][1]))
            ET.SubElement(bndbox, "xmax").text = str(int(annotation['bbox'][0] + annotation['bbox'][2]))
            ET.SubElement(bndbox, "ymax").text = str(int(annotation['bbox'][1] + annotation['bbox'][3]))

            # Draw bounding box and class name on the image
            class_name = next((cat['name'] for cat in categories if cat['id'] == annotation['category_id']), "Unknown")
            cv2.putText(image, class_name, 
                        (int(annotation['bbox'][0]), int(annotation['bbox'][1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(image, 
                          (int(annotation['bbox'][0]), int(annotation['bbox'][1])), 
                          (int(annotation['bbox'][0] + annotation['bbox'][2]), int(annotation['bbox'][1] + annotation['bbox'][3])), 
                          (0, 255, 0), 2)

        # Convert XML to a pretty string
        xml_str = minidom.parseString(ET.tostring(annotation_xml)).toprettyxml(indent="   ")

        # Determine if the image is in train or val set
        if image_id in train_image_id_to_filename:
            output_dir = train_annotations_dir
            annotated_image_path = os.path.join(train_annotated_dir, image_filename)
            shutil.copy(image_path, os.path.join(train_images_dir, image_filename))
        else:
            output_dir = val_annotations_dir
            annotated_image_path = os.path.join(val_annotated_dir, image_filename)
            shutil.copy(image_path, os.path.join(val_images_dir, image_filename))

        # Save the annotated image
        cv2.imwrite(annotated_image_path, image)
        xml_filename = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}.xml")
        with open(xml_filename, "w") as xml_file:
            xml_file.write(xml_str)

    # Repeat the process for validation set
    for annotation in coco_annotations['annotations']:
        image_id = annotation['image_id']
        if image_id in val_image_id_to_filename:
            image_filename = val_image_id_to_filename[image_id]
            image_info = next(item for item in val_images if item['id'] == image_id)

            # Create Pascal VOC XML structure
            annotation_xml = ET.Element("annotation")
            ET.SubElement(annotation_xml, "folder").text = "images"
            ET.SubElement(annotation_xml, "filename").text = image_filename

            size = ET.SubElement(annotation_xml, "size")
            ET.SubElement(size, "width").text = str(image_info['width'])
            ET.SubElement(size, "height").text = str(image_info['height'])
            ET.SubElement(size, "depth").text = "3"  # Assuming RGB images

            obj = ET.SubElement(annotation_xml, "object")
            ET.SubElement(obj, "name").text = str(annotation['category_id'])
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(annotation['bbox'][0]))
            ET.SubElement(bndbox, "ymin").text = str(int(annotation['bbox'][1]))
            ET.SubElement(bndbox, "xmax").text = str(int(annotation['bbox'][0] + annotation['bbox'][2]))
            ET.SubElement(bndbox, "ymax").text = str(int(annotation['bbox'][1] + annotation['bbox'][3]))

            # Convert XML to a pretty string
            xml_str = minidom.parseString(ET.tostring(annotation_xml)).toprettyxml(indent="   ")

            # Copy image to the validation JPEGImages directory
            shutil.copy(os.path.join(coco_images_dir, image_filename), os.path.join(val_images_dir, image_filename))

            # Draw bounding box and class name on the image
            class_name = next((cat['name'] for cat in categories if cat['id'] == annotation['category_id']), "Unknown")
            cv2.putText(image, class_name, 
                        (int(annotation['bbox'][0]), int(annotation['bbox'][1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            image_path = os.path.join(coco_images_dir, image_filename)
            image = cv2.imread(image_path)
            cv2.rectangle(image, 
                          (int(annotation['bbox'][0]), int(annotation['bbox'][1])), 
                          (int(annotation['bbox'][0] + annotation['bbox'][2]), int(annotation['bbox'][1] + annotation['bbox'][3])), 
                          (0, 255, 0), 2)
            
            # Save the annotated image
            annotated_image_path = os.path.join(val_annotated_dir, image_filename)
            cv2.imwrite(annotated_image_path, image)
            xml_filename = os.path.join(val_annotations_dir, f"{os.path.splitext(image_filename)[0]}.xml")
            with open(xml_filename, "w") as xml_file:
                xml_file.write(xml_str)

# Example usage
convert_coco_to_pascal("augmented-dataset/result.json", "augmented-dataset/images", "pascal_voc_dataset")
