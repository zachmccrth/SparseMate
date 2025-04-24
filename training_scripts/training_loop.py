import torch
from tqdm import tqdm

from training_scripts.train_event_logging import EventLogger, HasLoggableEvents, log_event


class TrainingLoop(HasLoggableEvents):
    """
    Responsible for defining the flow of training.

    Includes:
        Calling operations during training loop
        Device used in training
    """
    def __init__(self, model, training_data, steps ,optimizer, scheduler, criterion, logger, device):
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
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.TOTAL_STEPS = steps
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
        sparse_reconstruction = self.model(activations)
        loss = self.criterion(sparse_reconstruction, activations)

        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        self._step += 1

        return loss

    def training_loop(self):
        for _ in tqdm(range(self.TOTAL_STEPS), total=self.TOTAL_STEPS, desc="Training"):
            activations = next(self.training_data)
            loss = self._training_step(activations)




