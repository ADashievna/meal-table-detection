import yaml
import os

CONFIG_PATH = os.getenv('CONFIG_PATH', 'config.yaml')

with open(CONFIG_PATH, 'r') as f:
    _config = yaml.safe_load(f)

PLATE_STATE_IOA_THRESHOLD = _config.get('plate_state_ioa_threshold', [])
PLATE_CLASSES = _config['plate_classes']
FOOD_CLASSES = _config['food_classes']
PLATE_STATE_CLASSES = _config.get('plate_state_classes', [])
MODEL_PATH = _config['model_path']
VIDEO_PATH = _config.get('video_path', None)
DATASET_PATH = _config.get('dataset_path', None)
DATASET_YAML_PATH = _config.get('dataset_yaml_path', None)