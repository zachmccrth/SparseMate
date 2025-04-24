import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loguru import logger
import json
from datetime import datetime

import torch
from dotenv import load_dotenv

from training_scripts.training_configuration import load_training_config
from training_scripts.training_loop import TrainingLoop


def initialize_training(training_config):
    dataset = training_config['dataset']()

    activations_generator = training_config['activations_generator'](
        layers=training_config['layers'],
        device=training_config['device'],
        dtype=training_config['dtype'],
    )
    activations_buffer = training_config['activations_buffer'](
        dataset=dataset,
        activations_generator=activations_generator,
        buffer_size=training_config['buffer_size'],
        device=training_config['device'],
        dtype=training_config['dtype'],
    )

    model = training_config['model'](
        basis_size=training_config['basis_size'],
        embedding_dimensions=training_config['embedding_size'],
        iterations=training_config['layers'],
        device=training_config['device'],
        dtype=training_config['dtype'],
    )

    training_loop = TrainingLoop(
        model=model,
        training_data=activations_buffer,
        steps=training_config['steps'],
        batch_size=training_config['batch_size'],
        optimizer=training_config['optimizer'](params=model.parameters(), lr=training_config['lr']),
        scheduler=None,
        criterion=training_config['criterion'](),
        logger=training_config['logger'](save_dir=save_dir),
        device=training_config['device'],
    )
    return training_loop


if __name__ == '__main__':
    load_dotenv()

    logger.remove()
    logger.add(sys.stdout, level="INFO")

    training_config = load_training_config()

    save_dir = training_config.get('save_dir')
    if not save_dir:
        run_name = f"{datetime.now().strftime('%m%d_%H:%M')}_{training_config['model'].__class__.__name__}"
        save_dir = os.path.join(os.getenv("RUNS_DIR"), run_name)
        training_config['save_dir'] = save_dir
    os.makedirs(save_dir)

    # Save training configuration
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(training_config, f, indent=4, default=str)

    training_loop: TrainingLoop = initialize_training(training_config)
    
    model = training_loop.run()

    torch.save(obj=model, f = os.path.join(save_dir, "ae.pt"))
    

    
    
    


