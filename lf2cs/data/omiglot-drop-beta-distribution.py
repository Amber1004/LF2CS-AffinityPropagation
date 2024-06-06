import os
import numpy as np
from PIL import Image
import random
import shutil
import json

def load_omniglot(data_dir):
    """ Load Omniglot data from directories """
    data_dict = {'data': [], 'labels': []}
    label_count = 0
    for label_dir in sorted(os.listdir(data_dir)):
        full_dir = os.path.join(data_dir, label_dir)
        if os.path.isdir(full_dir):
            for img_file in os.listdir(full_dir):
                img_path = os.path.join(full_dir, img_file)
                data_dict['data'].append(img_path)
                data_dict['labels'].append(label_count)
            label_count += 1
    return data_dict

def generate_drop_rates_beta(num_classes, alpha=0.2, beta=0.5, min_rate=0.1, max_rate=0.8):
    """Generate drop rates using a beta distribution, scaled to fluctuate within a given range."""
    rates = np.random.beta(alpha, beta, num_classes)
    scaled_rates = min_rate + rates * (max_rate - min_rate)  # Scale rates to the range 0.1 to 0.8
    return scaled_rates

def drop_images(data_dict, drop_rates, min_images=1):
    """ Randomly drop images from each class in the dataset with varying drop rates while ensuring a minimum number of images in each class """
    new_data = []
    new_labels = []
    drop_rate_list = []

    for cls in set(data_dict['labels']):
        cls_indices = [i for i, label in enumerate(data_dict['labels']) if label == cls]
        original_count = len(cls_indices)
        drop_rate = drop_rates[cls]
        num_to_drop = int(drop_rate * original_count)
        num_to_keep = original_count - num_to_drop
        
        if num_to_keep < min_images:
            num_to_keep = min_images
            drop_rate = 1 - (num_to_keep / original_count)  # Adjust drop rate accordingly
        
        kept_indices = random.sample(cls_indices, num_to_keep)
        new_data.extend([data_dict['data'][i] for i in kept_indices])
        new_labels.extend([data_dict['labels'][i] for i in kept_indices])
        drop_rate_list.append((cls, drop_rate))

    save_drop_rates(drop_rate_list, 'adjusted_drop_rates.json')
    return {'data': new_data, 'labels': new_labels}

def save_drop_rates(drop_rates, file_path):
    """ Save the drop rates to a JSON file """
    drop_rate_dict = {f"Class_{cls}": rate for cls, rate in drop_rates}
    with open(file_path, 'w') as json_file:
        json.dump(drop_rate_dict, json_file, indent=4)

def unpack_images_to_folders(data_dict, output_dir, set_name):
    """ Save images to new directories based on their class labels and set type """
    for img_path, label in zip(data_dict['data'], data_dict['labels']):
        label_dir = os.path.join(output_dir, set_name, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        dest_path = os.path.join(label_dir, os.path.basename(img_path))
        shutil.copy(img_path, dest_path)

# Main execution
data_dirs = {
    'train': 'data/omniglot/train',
    'test': 'data/omniglot/test',
    'val': 'data/omniglot/val'
}
output_dir = 'data/imbalanced_omniglot'

for set_name, dir_path in data_dirs.items():
    data_dict = load_omniglot(dir_path)
    num_classes = max(data_dict['labels']) + 1
    drop_rates = generate_drop_rates_beta(num_classes)
    data_dict = drop_images(data_dict, drop_rates)
    unpack_images_to_folders(data_dict, output_dir, set_name)
