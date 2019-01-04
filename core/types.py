from enum import Enum


class StrEnum(str, Enum):
    pass


class EpochPhase(StrEnum):
    Training = "train"
    Evaluating = "val"


class TorchDevice(StrEnum):
    Cuda = "cuda:0"
    Cpu = "cpu"


class ModelName(StrEnum):
    Resnet = "resnet"
    Alexnet = "alexnet"
    Vgg = "vgg"
    Squeezenet = "squeezenet"
    Densenet = "densenet"
    Inception = "inception"
