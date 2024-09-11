import os

import yaml


def get_config():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath("")))
    with open(os.path.join(project_dir, "config.yaml")) as f:
        return yaml.load(f, Loader=yaml.FullLoader)
