DEFAULT_BSNN_CONFIG_VALUES = {
    # Model Architecture
    'num_blocks': 2,        
    'base_width': 8,
    'base_factor': 2,
    'pool_kernel_size': 2,
    'pool_stride_size': 2,
    'conv_kernel_size': 5,
    'conv_stride_size': 1,
    'conv_padding_size': 0,
    'conv_dilation_size': 1,
    
    # Image Properties
    'num_classes': 10,
    'img_height': 32,
    "img_width": 32,
    'img_num_frames': 32,
    'frame_filter_threshold': 10000,

    
    # Training Parameters
    'num_epochs': 500,
    'num_trials': 100,
    'num_steps': 100,
    
    # Hyperparameter Sweep
    'beta_value': 0.41628667820177095,
    'beta_range': [0.9, 0.99],
    'thresholds': [22.37297929074331, 1.738555820352834, 19.083022150261986],
    'dropout': 0.48143584931131816,
    'slope': 0.9870918226886212,
    # Loss Function
    "correct_rate": 0.8,
    "incorrect_rate": 0.2,
    "is_batch_norm": False,
}


DEFAULT_LOADER_CONFIG_VALUES = {
    'data_path': './train/DVSGesture',
    'data_cache_path': './cache/DVSGesture',
    'label_csv_path': './train/DVSGesture/gesture_mapping.csv',
    'batch_size': 32,
    'use_data_cache': True,
}


def prepare_config(current_config, default_config):
    if current_config is None:
        return default_config.copy()
    
    prepared_config = current_config.copy()
    
    for key, value in default_config.items():
        prepared_config.setdefault(key, value)
        
    return prepared_config