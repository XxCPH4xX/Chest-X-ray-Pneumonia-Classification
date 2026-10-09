from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Scores(BaseModel):
    NORMAL: float = Field(ge=0, le=1)
    PNEUMONIA: float = Field(ge=0, le=1)


class Images(BaseModel):
    model_input: str
    heatmap: str
    overlay: str


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_id: str
    created_at: str
    predicted_class: Literal["NORMAL", "PNEUMONIA", "INCONCLUSIVE"]
    probabilities: Scores
    threshold: float
    architecture: str
    model_version: str
    gradcam_available: bool
    gradcam_target: str
    uncertainty_policy: dict | None
    seconds: float
    saved: bool
    images: Images | None = None
