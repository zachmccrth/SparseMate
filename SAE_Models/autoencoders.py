import json
import os

class AutoEncoderDirectory:

    def __init__(self):
        self.current_directory = os.path.dirname(os.path.abspath(__file__))
        self.models = None
        self.refresh_models_list()


    def refresh_models_list(self):
        """
        Refreshes the list of models within the autoencoder directory.

        Pulls configuration information from the config.json file associated with each model for later use
        """
        self.models = dict()

        # Iterate through directories in the current directory
        for item in os.listdir(self.current_directory):
            item_path = os.path.join(self.current_directory, item)

            # Only want directories with config files (model directories)
            if not(os.path.isdir(item_path)):
                continue
            else:
                config_path = os.path.join(item_path, "config.json")
                if not (os.path.isfile(config_path)):
                    continue


            try:
                with open(config_path) as json_file:
                    config = json.load(json_file)
                    config["timestamp"] = os.path.getctime(config_path)
                    config["model_location"] = os.path.join(item_path, "ae.pt")
                    self.models[item] = config


            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {config_path}: {e}")
            except Exception as e:
                print(f"Error reading {config_path}: {e}")

