from enum import Enum


class EDeviceType(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"
