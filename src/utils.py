# utils.py

import os
import logging
import json

def get_env_variable(var_name, default_value=None):
    """
    Retrieves an environment variable.
    :param var_name: Name of the environment variable.
    :param default_value: Default value if the environment variable is not found.
    :return: Value of the environment variable.
    """
    return os.getenv(var_name, default_value)

def setup_logging(default_path='GCP_key.json', default_level=logging.INFO, env_key='LOG_CFG'):
    """
    Setup logging configuration.
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
