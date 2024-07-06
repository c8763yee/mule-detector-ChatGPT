from typing import Literal

from pydantic import BaseModel


class OpenAIConfig(BaseModel):
    max_tokens: int = 1024
    model: str = "gpt-4o"
    temperature: float = 0


class VisionConfig:
    detail: Literal["low", "high", "auto"] = "low"


class Config(BaseModel):
    DEBUG: bool = True
