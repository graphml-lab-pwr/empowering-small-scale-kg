from typing import Type, Union

from pykeen.models import RotatE

T_MODEL = Union[RotatE]
T_MODEL_TYPE = Union[Type[RotatE]]


def get_model(model_name: str) -> T_MODEL_TYPE:
    if model_name == "RotatE":
        return RotatE
    else:
        raise ValueError
