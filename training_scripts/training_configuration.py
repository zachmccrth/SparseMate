import json
import os
from dotenv import load_dotenv

from training_scripts.data_embedding_map import TruncatedLeelaDataEmbeddingMap

load_dotenv()
if __name__ == "__main__":

    training_config_name = "test.json"
    training_config_dir = os.getenv("TRAINING_CONFIG_DIR")

    if training_config_name:
        training_config_path = os.path.join(training_config_dir, training_config_name)
        with open(training_config_path, "r") as f:
            training_config = json.load(f)
            print(f"Using training config: {training_config_name}")

    if not training_config:
        training_config = {
            'activations_generator_class': TruncatedLeelaDataEmbeddingMap

        }

