from enum import Enum


class ECriterion(str, Enum):
    CE = "ce"
    BCE = "bce"
    BCEL = "bcel"
