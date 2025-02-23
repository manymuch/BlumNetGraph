from .skeleton import DATASET_OPTIONS
from .skeleton import build as build_skeleton


def build_dataset(image_set, config):
    if config["dataset_file"] in DATASET_OPTIONS:
        return build_skeleton(image_set, config)
    else:
        raise ValueError(f'dataset {config["dataset_file"]} not supported')
