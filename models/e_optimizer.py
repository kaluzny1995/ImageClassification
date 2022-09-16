from enum import Enum


class EOptimizer(str, Enum):
    ADAM = "adam"
    ADAMAX = "adamax"
    ADAGRAD = "adagrad"
