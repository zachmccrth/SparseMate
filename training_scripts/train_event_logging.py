from __future__ import annotations
from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, Callable, Protocol, TypeVar, Union, Mapping, runtime_checkable
from functools import wraps


@runtime_checkable
class EventLogger(Protocol):
    def log_event(self, result: Any, tag: str, step: int) -> None:
        ...


class TensorboardEventLogger(EventLogger):
    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter
        self.logger: SummaryWriter = SummaryWriter(log_dir)

    def log_event(self, result: Any, tag: str, step: int) -> None:
        if isinstance(result, Number):
            self.logger.add_scalar(tag, result, step)

        elif isinstance(result, Mapping):
            for key, val in result.items():
                if isinstance(val, Number):
                    self.logger.add_scalar(f"{tag}/{key}", val, step)
                # TODO: Extend for other types

        elif hasattr(result, "log") and callable(getattr(result, "log")):
            result.log(self, tag, step)

        else:
            raise TypeError(f"Unsupported event type: {type(result)}")



class HasLoggableEvents(ABC):
    @property
    @abstractmethod
    def logger(self) -> EventLogger:
        ...

    @property
    @abstractmethod
    def step(self) -> int:
        ...


F = TypeVar("F", bound=Callable[..., Any])

def log_event(tag: str | None = None) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self: HasLoggableEvents, *args: Any, **kwargs: Any) -> Any:
            if not hasattr(self, "logger"):
                raise AttributeError(
                    f"{self.__class__.__name__} is missing required property 'logger'."
                )
            if not hasattr(self, "step"):
                raise AttributeError(
                    f"{self.__class__.__name__} is missing required property 'step'."
                )
            result = func(self, *args, **kwargs)
            actual_tag = tag or func.__name__
            self.logger.log_event(result, actual_tag, self.step)
            return result
        return wrapper  # type: ignore
    return decorator
