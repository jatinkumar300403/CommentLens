# register model

import json
import mlflow
import logging
import os
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
MODEL_NAME = 'yt_chrome_plugin_model'
MODEL_ALIAS = 'staging'
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))


# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)

        # Model stages are deprecated since MLflow 2.9; an alias marks the version to serve
        client = MlflowClient()
        client.set_registered_model_alias(model_name, MODEL_ALIAS, model_version.version)

        logger.debug(f'Model {model_name} version {model_version.version} registered with alias "{MODEL_ALIAS}".')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        model_info = load_model_info(os.path.join(ROOT_DIR, 'experiment_info.json'))
        register_model(MODEL_NAME, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        raise


if __name__ == '__main__':
    main()
