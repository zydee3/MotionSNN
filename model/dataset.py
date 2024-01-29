import tonic
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader


DEFAULT_LOADER_CONFIG_VALUES = {
    'data_path': './train/DVSGesture',
    'data_cache_path': './cache/DVSGesture',
    'label_csv_path': './train/DVSGesture/gesture_mapping.csv',
    'batch_size': 32,
    'use_data_cache': True,
}

def prepare_loader_config(config):
    if config is None:
        return DEFAULT_LOADER_CONFIG_VALUES.copy()
    
    prepared_config = config.copy()
    
    for key, value in DEFAULT_LOADER_CONFIG_VALUES.items():
        prepared_config.setdefault(key, value)
        
    return prepared_config


def load_data_loader(data_path, data_cache_path, batch_size, cache):
    dataset = tonic.datasets.DVSGesture(save_to=data_path)
    
    if cache:
        cached_trainset = DiskCachedDataset(dataset, cache_path=data_cache_path)
        return DataLoader(cached_trainset)
    else:
        return DataLoader(dataset)


def load_data_labels(label_csv_path):
    labels = []
    
    try:
        dataset_map_file = open(label_csv_path, "r")
        dataset_map_file.readline()
        
        for line in dataset_map_file:
            gesture, label = line.strip().split(',')
            label = int(label)
            
            while len(labels) <= label:
                labels.append(None)
            
            labels[label] = gesture
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {label_csv_path}")
    
    return labels


def load(config):
    config = prepare_loader_config(config)
    
    loader = load_data_loader(
        config['data_path'],
        config['data_cache_path'],
        config['batch_size'],
        config['use_data_cache'],
    )
    
    labels = load_data_labels(
        config['label_csv_path']
    )
    
    return loader, labels