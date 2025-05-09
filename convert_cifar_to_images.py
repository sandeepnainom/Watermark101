import pickle
import os
import numpy as np
from PIL import Image

# Helper: Load a batch file
def load_cifar_batch(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return images, labels

# Helper: Load label names from batches.meta
def load_label_names(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
        return [label.decode('utf-8') for label in meta[b'label_names']]

# Paths
data_dir = os.path.join(os.getcwd(), 'data', 'original_images', 'cifar-10-batches-py')

output_dir = 'C:\\cnn_watermarking_project\data1\images'
os.makedirs(output_dir, exist_ok=True)

# Load class names
label_names = load_label_names(os.path.join(data_dir, 'batches.meta'))

# Process each data_batch_1 to data_batch_5
image_count = 0
for i in range(1, 6):
    batch_file = os.path.join(data_dir, f'data_batch_{i}')
    images, labels = load_cifar_batch(batch_file)
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        class_name = label_names[label]
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(class_dir, f"{class_name}_{image_count + 1}.png"))
        image_count += 1

print(f"Saved {image_count} images in folders by class at: {output_dir}")
