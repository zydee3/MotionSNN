DEFAULT_BSNN_CONFIG_VALUES = {
    'num_steps': 15,
    'num_epochs': 50,
    'learning_rate': 0.0001,
    'thresholds': [2, 2, 3],
    'activation_threshold': 0.4,
}

def prepare_config(config):
    if config is None:
        return DEFAULT_BSNN_CONFIG_VALUES.copy()
    
    prepared_config = config.copy()
    
    for key, value in DEFAULT_BSNN_CONFIG_VALUES.items():
        prepared_config.setdefault(key, value)
        
    return prepared_config
