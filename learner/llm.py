from __future__ import annotations

import json
import os
import subprocess
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str: ...


class ClaudeCliBackend(LLMBackend):
    """Calls the 'claude' CLI subprocess and returns its stdout."""

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.model = model

    def complete(self, prompt: str) -> str:
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", self.model],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()


class MockLLM(LLMBackend):
    """Returns deterministic canned responses for offline testing."""

    _CONTENT_JSON = json.dumps({
        "summary": "Mock summary for offline testing.",
        "cards": [
            {"front": "What is spaced repetition?", "back": "A learning technique that uses increasing intervals between reviews."},
        ],
        "questions": [
            {
                "stem": "What does spaced repetition help with?",
                "choices": ["Memory retention", "Speed reading", "Note-taking", "Summarising"],
                "answer_index": 0,
                "explanation": "Spaced repetition optimises long-term memory retention.",
            }
        ],
    })

    def complete(self, prompt: str) -> str:
        if "study coach" in prompt:
            return "Mock study plan: review your weak topics for 30 minutes daily."
        return self._CONTENT_JSON


# Backwards-compat alias
MockBackend = MockLLM


def get_backend(backend: str | bool = "auto") -> LLMBackend:
    """Return a backend by name ("mock", "claude") or legacy bool flag."""
    if backend == "mock" or backend is True or os.environ.get("LEARNER_MOCK"):
        return MockLLM()
    return ClaudeCliBackend()
