from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class LLMProvider(Enum):
    OPENAI = "openai"
    OPENVINO = "openvino"
    MOCK = "mock"


@dataclass
class LLMRequest:
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 512


@dataclass
class LLMResponse:
    text: str
    raw: Dict[str, Any] | None = None


class LLMClient:
    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError("Subclasses must implement generate()")
