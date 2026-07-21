from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ChatMessage:
    role: Role
    content: str
    name: Optional[str] = None

    def as_dict(self) -> dict[str, str]:
        payload = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        return payload


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    raw: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)


class LLM(ABC):
    """Common interface for text-generation providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        raise NotImplementedError

    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        prompt_parts = []
        system_prompt = None
        for message in messages:
            if message.role == "system":
                system_prompt = message.content
            else:
                prompt_parts.append(f"{message.role.upper()}: {message.content}")
        return self.generate(
            "\n\n".join(prompt_parts),
            system_prompt=system_prompt,
            **kwargs,
        )
