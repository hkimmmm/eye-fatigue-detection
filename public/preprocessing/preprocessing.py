import cv2
import numpy as np
import os

def preprocess_image(image_path, target_size=(64, 64)):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be read.")
    # Resize the image
    image_resized = cv2.resize(image, target_size)
    # Normalize pixel values to [0, 1]
    image_normalized = image_resized / 255.0
    return image_normalized

def preprocess_dataset(dataset_path):
    images = []
    labels = []
    label_mapping = {}  # Dictionary untuk mapping label string ke angka

    for idx, label in enumerate(os.listdir(dataset_path)):
        label_path = os.path.join(dataset_path, label)
        
        # Buat mapping label string ke angka
        if label not in label_mapping:
            label_mapping[label] = idx
        
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            image = preprocess_image(image_path)
            images.append(image)
            labels.append(label_mapping[label])  # Simpan angka, bukan string

    print(f"Label Mapping: {label_mapping}")  # Debugging untuk melihat mapping
    return np.array(images), np.array(labels)

# Example usage
if __name__ == "__main__":
    images, labels = preprocess_dataset('data/train')
    print(f"Processed {len(images)} images and {len(set(labels))} labels.")
