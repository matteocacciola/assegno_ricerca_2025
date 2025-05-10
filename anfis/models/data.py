from typing import Any, Callable
import numpy as np
from pydantic import BaseModel, model_validator, field_validator


class FractionData(BaseModel):
    fraction: float | None = None
    fraction_callback: Any = None

    @field_validator("fraction")
    def check_fraction(cls, value):
        if value is None:
            return value
        value = float(value)
        if value == 0 or value > 1:
            raise ValueError("fraction must be between 0 and 1")
        return value

    @field_validator("fraction_callback")
    def check_fraction_callback(cls, value):
        if value is None:
            return value
        if not callable(value):
            raise ValueError("fraction_callback must be a callable function")
        return value

    @model_validator(mode="after")
    def check_model(self):
        if self.fraction is not None and self.fraction < 1 and self.fraction_callback is None:
            raise ValueError("fraction_callback must be provided if fraction < 1")
        return self


class PredictionParser(BaseModel):
    parser: Callable[[np.ndarray, int | None], np.ndarray]
    n_classes: int | None = None
