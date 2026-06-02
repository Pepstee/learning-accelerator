from __future__ import annotations

import subprocess
from typing import Protocol


class LLMBackend(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class ClaudeCliBackend:
    """Calls the 'claude' CLI subprocess and returns its stdout."""

    def __init__(self, model: str = "claude-sonnet-4-5") -> None:
        self.model = model

    def complete(self, prompt: str) -> str:
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", self.model],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()


class MockBackend:
    """Returns deterministic canned JSON for offline testing."""

    _RESPONSE = (
        '{"cards": [], "questions": [], "weak_areas": []}'
    )

    def complete(self, prompt: str) -> str:
        return self._RESPONSE
