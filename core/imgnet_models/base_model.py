from abc import ABC, abstractmethod

from core.types import ModelName


class ImgnetModel(ABC):
    @abstractmethod
    def init(
            self,
            device,
            num_classes: int,
            feature_extract: bool,
            use_pretrained: bool = True
    ) -> None:
        pass

    @property
    @abstractmethod
    def model_name(self) -> ModelName:
        pass

    @property
    @abstractmethod
    def model_ft(self):
        pass

    @property
    @abstractmethod
    def input_size(self) -> int:
        pass
