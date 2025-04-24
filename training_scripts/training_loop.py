import torch
from tqdm import tqdm

from training_scripts.activation_buffer import ActivationBuffer
from training_scripts.train_event_logging import EventLogger, HasLoggableEvents, log_event


class TrainingLoop(HasLoggableEvents):
    """
    Responsible for defining the flow of training.

    Includes:
        Calling operations during training loop
        Device used in training
    """
    def __init__(self, model, training_data: ActivationBuffer, steps, batch_size ,optimizer, scheduler, criterion, logger, device):
        """
        model: model to train
        training_data: iterable training data
        optimizer: calculates the gradient of the loss with respect to the model's parameters
        scheduler: adjusts parameters of the optimizer during training
        criterion: loss function
        device: cpu or gpu
        """
        self.model = model
        self.training_data = training_data
        self.TOTAL_STEPS = steps
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device


        self._logger = logger
        self._step = 0


    @property
    def logger(self) -> EventLogger:
        if self._logger is None:
            raise AttributeError("Logger not initialized")
        return self._logger

    @property
    def step(self) -> int:
        return self._step

    @log_event("train_loss")
    def _training_step(self, activations: torch.Tensor):
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        residuals = self.model(activations)
        loss = self.criterion(residuals, activations)

        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        self._step += 1

        return loss.item()

    def run(self):
        total_tokens = self.batch_size * self.TOTAL_STEPS

        progress_bar = tqdm(total=total_tokens, desc="Training", unit="tokens")

        for _ in range(self.TOTAL_STEPS):
            activations = self.training_data.load_next_data(self.batch_size)
            loss = self._training_step(activations)

            progress_bar.update(self.batch_size)
        progress_bar.close()
        return self.model





