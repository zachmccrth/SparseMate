from typing import List
from torch.utils.data import Dataset

class DatasetsAPI:
    """
    A registry class for accessing imported datasets
    """
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(dataset_cls):
            cls._registry[name] = dataset_cls
            return dataset_cls
        return decorator

    @classmethod
    def get(cls, name, *args, **kwargs) -> Dataset:
        if name not in cls._registry:
            raise ValueError(f"Dataset '{name}' is not registered.")
        return cls._registry[name](*args, **kwargs)

    @classmethod
    def list_datasets(cls) -> List[str]:
        return list(cls._registry.keys())