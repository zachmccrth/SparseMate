import os
from datetime import datetime
from dotenv import load_dotenv
import torch



from training_scripts.training_configuration import load_training_config
from training_scripts.training_loop import TrainingLoop

if __name__ == '__main__':
    load_dotenv()

    training_config = load_training_config()

    save_dir = training_config.get('save_dir')
    if not save_dir:
        run_name = f"{datetime.now().strftime('%m%d_%H:%M')}_{training_config['model']}"
        save_dir = os.path.join(os.getenv("RUNS_DIR"), run_name)
        training_config['save_dir'] = save_dir
    os.makedirs(save_dir)

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

    training_loop = TrainingLoop(
        model=training_config['model'](),
        training_data=activations_buffer,
        steps=training_config['steps'],
        batch_size=training_config['batch_size'],
        optimizer=training_config['optimizer'](lr=training_config['lr']),
        scheduler=None,
        criterion=training_config['criterion'](),
        logger=training_config['logger'](save_dir=save_dir),
        device=training_config['device'],
    )

