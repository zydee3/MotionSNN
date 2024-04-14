import tonic
import numpy as np
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader

from tonic.transforms import ToFrame
from tonic.datasets import DVSGesture


from model.config import prepare_config, DEFAULT_LOADER_CONFIG_VALUES

def load_dataset(config, is_training_set=True, transform=None):
    dataset = tonic.datasets.DVSGesture(
        save_to=config['data_path'], 
        transform=transform,
        train=is_training_set
    )
    
    return dataset


def load_data_labels(config):
    labels = []
    
    try:
        dataset_map_file = open(config['label_csv_path'], "r")
        dataset_map_file.readline()
        
        for line in dataset_map_file:
            gesture, label = line.strip().split(',')
            label = int(label)
            
            while len(labels) <= label:
                labels.append(None)
            
            labels[label] = gesture
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {config['label_csv_path']}")
    
    return labels


def normalize_frames(frames, lower=5, upper=95):
    lower_percentile = np.percentile(frames, lower)
    upper_percentile = np.percentile(frames, upper)
    
    if upper_percentile <= lower_percentile:
        upper_percentile = max(upper_percentile + 1e-3, lower_percentile + 1e-2)
    

    frames_normalized = (frames - lower_percentile) / (upper_percentile - lower_percentile)
    frames_normalized = np.clip(frames_normalized, 0, 1)  
    
    return frames_normalized


def to_frames(events):
     # creates dense frames from events by binning them in different ways
    frame_transform = ToFrame(
        sensor_size=DVSGesture.sensor_size,
        n_time_bins=100)
    
    return frame_transform(events)


def to_cache(config, dataset):
    if config['use_data_cache'] is False: 
        print("Warning: Data caching is disabled. Original dataset will be used.")
        return dataset
    else:
        return DiskCachedDataset(
            dataset,
            cache_path=config['data_cache_path'],
        )
    
    
def load(config, transform=None):
    config = prepare_config(config, DEFAULT_LOADER_CONFIG_VALUES)
    train = load_dataset(config=config, is_training_set=True, transform=transform)
    test = load_dataset(config=config, is_training_set=False, transform=transform)
    labels = load_data_labels(config=config)
    return train, test, labels
