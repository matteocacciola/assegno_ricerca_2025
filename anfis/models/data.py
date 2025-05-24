from typing import Callable
import pandas as pd
from pydantic import BaseModel, field_validator


class FractionData(BaseModel):
    fraction: float
    fraction_callback: Callable[[pd.DataFrame, float], pd.DataFrame]

    @field_validator("fraction")
    def check_fraction(cls, value):
        if value is None:
            return value
        value = float(value)
        if value == 0 or value > 1:
            raise ValueError("fraction must be between 0 and 1")
        return value
