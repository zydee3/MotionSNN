import tonic
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader


DEFAULT_LOADER_CONFIG_VALUES = {
    'data_path': './train/DVSGesture',
    'data_cache_path': './cache/DVSGesture',
    'label_csv_path': './train/DVSGesture/gesture_mapping.csv',
    'batch_size': 32,
    'use_data_cache': True,
    'transform': None,
}

def prepare_loader_config(config):
    if config is None:
        return DEFAULT_LOADER_CONFIG_VALUES.copy()
    
    prepared_config = config.copy()
    
    for key, value in DEFAULT_LOADER_CONFIG_VALUES.items():
        prepared_config.setdefault(key, value)
        
    return prepared_config


def load_data_loader(config, is_training_set):
    dataset = tonic.datasets.DVSGesture(
        save_to=config['data_path'], 
        train=is_training_set, 
        transform=config['transform']
    )
    
    if config['use_data_cache']:
        cached_trainset = DiskCachedDataset(
            dataset, 
            cache_path=config['data_cache_path']
        )
        
        return DataLoader(cached_trainset)
    else:
        return DataLoader(dataset)


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


def load(config):
    config = prepare_loader_config(config)  
    train_loader = load_data_loader(config=config, is_training_set=True)
    test_loader = load_data_loader(config=config, is_training_set=False)
    label_map = load_data_labels(config=config)
    return train_loader, test_loader, label_map