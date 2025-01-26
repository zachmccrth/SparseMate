import json
import os


current_directory = os.path.dirname(os.path.abspath(__file__))


# Iterate through directories in the current directory
for item in os.listdir(current_directory):
    item_path = os.path.join(current_directory, item)

    # Only want directories with config files (model directories)
    if not(os.path.isdir(item_path)):
        continue
    else:
        config_path = os.path.join(item_path, "config.json")
        if not (os.path.isfile(config_path)):
            continue
        else:
            models_directory = item_path

    try:
        with open(config_path) as json_file:
            config = json.load(json_file)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {config_path}: {e}")
    except Exception as e:
        print(f"Error reading {config_path}: {e}")

