"""
Adversarial tests for learner.llm — independent authorship.

Covers:
  - LLMBackend ABC contract (cannot instantiate; subclass must implement complete)
  - MockLLM canned-response contract (determinism, routing, JSON validity)
  - MockBackend backwards-compat alias
  - ClaudeCliBackend with a mocked subprocess (correct argv, stripping, error paths)
  - get_backend factory (all code paths including env-var override)

Nothing about the unit under test is itself mocked: the real classes are exercised.
ClaudeCliBackend's subprocess.run is patched so no real 'claude' binary is needed.
"""
from __future__ import annotations

import json
import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from learner.llm import (
    ClaudeCliBackend,
    LLMBackend,
    MockBackend,
    MockLLM,
    get_backend,
)


# ── LLMBackend abstract base class ──────────────────────────────────────────

class TestLLMBackendABC:
    """LLMBackend is abstract; concrete subclasses must implement complete()."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            LLMBackend()  # type: ignore[abstract]

    def test_subclass_without_complete_cannot_instantiate(self):
        class Incomplete(LLMBackend):
            pass  # forgot to implement complete()

        with pytest.raises(TypeError):
            Incomplete()

    def test_minimal_concrete_subclass_works(self):
        class Minimal(LLMBackend):
            def complete(self, prompt: str) -> str:
                return "ok"

        backend = Minimal()
        assert backend.complete("x") == "ok"

    def test_complete_signature_accepts_string_prompt(self):
        class Echo(LLMBackend):
            def complete(self, prompt: str) -> str:
                return prompt

        assert Echo().complete("hello") == "hello"

    def test_complete_must_return_str(self):
        # The ABC cannot enforce the return type at runtime, but we verify the
        # contract by checking a conforming subclass — tests remain meaningful
        # because they catch regressions in the real implementations below.
        class Good(LLMBackend):
            def complete(self, prompt: str) -> str:
                return "result"

        assert isinstance(Good().complete("p"), str)


# ── MockLLM canned-response contract ─────────────────────────────────────────

class TestMockLLMCompleteReturnsString:
    def test_returns_str_for_generic_prompt(self):
        assert isinstance(MockLLM().complete("anything"), str)

    def test_returns_str_for_study_coach_prompt(self):
        assert isinstance(MockLLM().complete("You are a study coach"), str)

    def test_returns_str_for_empty_prompt(self):
        assert isinstance(MockLLM().complete(""), str)


class TestMockLLMContentJsonPath:
    """Prompts that do NOT contain 'study coach' must return valid, structured JSON."""

    def setup_method(self):
        self.backend = MockLLM()
        self.data = json.loads(self.backend.complete("generate flashcards"))

    def test_response_is_valid_json(self):
        # json.loads already ran in setup; if it raised, setup would fail
        assert isinstance(self.data, dict)

    def test_json_has_summary_key(self):
        assert "summary" in self.data

    def test_summary_is_non_empty_string(self):
        assert isinstance(self.data["summary"], str)
        assert self.data["summary"].strip()

    def test_json_has_cards_key(self):
        assert "cards" in self.data

    def test_cards_is_a_list(self):
        assert isinstance(self.data["cards"], list)

    def test_cards_list_not_empty(self):
        assert len(self.data["cards"]) >= 1

    def test_each_card_has_front(self):
        for i, card in enumerate(self.data["cards"]):
            assert "front" in card, f"card[{i}] missing 'front'"

    def test_each_card_has_back(self):
        for i, card in enumerate(self.data["cards"]):
            assert "back" in card, f"card[{i}] missing 'back'"

    def test_each_card_front_is_non_empty_string(self):
        for i, card in enumerate(self.data["cards"]):
            assert isinstance(card["front"], str) and card["front"].strip(), f"card[{i}].front"

    def test_each_card_back_is_non_empty_string(self):
        for i, card in enumerate(self.data["cards"]):
            assert isinstance(card["back"], str) and card["back"].strip(), f"card[{i}].back"

    def test_json_has_questions_key(self):
        assert "questions" in self.data

    def test_questions_is_a_list(self):
        assert isinstance(self.data["questions"], list)

    def test_questions_list_not_empty(self):
        assert len(self.data["questions"]) >= 1

    def test_each_question_has_stem(self):
        for i, q in enumerate(self.data["questions"]):
            assert "stem" in q, f"question[{i}] missing 'stem'"

    def test_each_question_has_choices(self):
        for i, q in enumerate(self.data["questions"]):
            assert "choices" in q, f"question[{i}] missing 'choices'"

    def test_each_question_has_answer_index(self):
        for i, q in enumerate(self.data["questions"]):
            assert "answer_index" in q, f"question[{i}] missing 'answer_index'"

    def test_each_question_has_explanation(self):
        for i, q in enumerate(self.data["questions"]):
            assert "explanation" in q, f"question[{i}] missing 'explanation'"

    def test_answer_index_in_range(self):
        for i, q in enumerate(self.data["questions"]):
            idx = q["answer_index"]
            assert 0 <= idx < len(q["choices"]), f"question[{i}].answer_index out of range"


class TestMockLLMStudyCoachPath:
    """Prompts containing 'study coach' must NOT return the content JSON."""

    def test_study_coach_returns_string(self):
        result = MockLLM().complete("You are a study coach, give a plan")
        assert isinstance(result, str)

    def test_study_coach_result_differs_from_content_json(self):
        backend = MockLLM()
        study_result = backend.complete("study coach")
        generic_result = backend.complete("generate cards")
        assert study_result != generic_result

    def test_study_coach_result_not_json(self):
        result = MockLLM().complete("study coach please")
        with pytest.raises((json.JSONDecodeError, ValueError)):
            json.loads(result)

    def test_study_coach_anywhere_in_prompt_triggers_path(self):
        result = MockLLM().complete("Act as a study coach and help me")
        # Should not be the content JSON
        assert result != MockLLM()._CONTENT_JSON

    def test_prompt_case_matters(self):
        # 'STUDY COACH' (uppercase) must NOT trigger the study-coach path
        upper_result = MockLLM().complete("STUDY COACH")
        content_json = MockLLM()._CONTENT_JSON
        # Either it's the content JSON or it's a different response — as long as
        # it doesn't trigger only for the exact lowercase match, we document it:
        # the implementation checks for lowercase "study coach".
        assert upper_result == content_json  # case-sensitive; uppercase → content JSON


class TestMockLLMDeterminism:
    """MockLLM must be purely deterministic — same prompt → same response, every time."""

    def test_same_generic_prompt_same_result(self):
        backend = MockLLM()
        assert backend.complete("hello") == backend.complete("hello")

    def test_different_generic_prompts_same_result(self):
        backend = MockLLM()
        # All non-study-coach prompts return the same canned JSON
        assert backend.complete("prompt A") == backend.complete("prompt B")

    def test_multiple_instances_return_same_content_json(self):
        assert MockLLM().complete("x") == MockLLM().complete("x")

    def test_content_json_class_attribute_matches_return_value(self):
        backend = MockLLM()
        assert backend.complete("generic") == backend._CONTENT_JSON

    def test_study_coach_result_stable(self):
        backend = MockLLM()
        assert backend.complete("study coach") == backend.complete("study coach")


class TestMockLLMIsLLMBackend:
    def test_isinstance_check(self):
        assert isinstance(MockLLM(), LLMBackend)

    def test_complete_method_present(self):
        assert callable(MockLLM().complete)


# ── MockBackend alias ─────────────────────────────────────────────────────────

class TestMockBackendAlias:
    """MockBackend must be exactly the same class as MockLLM (backwards compat)."""

    def test_alias_is_same_class(self):
        assert MockBackend is MockLLM

    def test_alias_instances_are_mock_llm(self):
        assert isinstance(MockBackend(), MockLLM)

    def test_alias_complete_returns_same_as_mockllm(self):
        assert MockBackend().complete("x") == MockLLM().complete("x")


# ── ClaudeCliBackend — subprocess mocking ────────────────────────────────────

def _make_completed_process(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.stdout = stdout
    cp.returncode = returncode
    return cp


class TestClaudeCliBackendInit:
    def test_default_model_is_claude_sonnet(self):
        backend = ClaudeCliBackend()
        assert backend.model == "claude-sonnet-4-6"

    def test_custom_model_stored(self):
        backend = ClaudeCliBackend(model="claude-opus-4-8")
        assert backend.model == "claude-opus-4-8"

    def test_is_llm_backend(self):
        assert isinstance(ClaudeCliBackend(), LLMBackend)


class TestClaudeCliBackendArgv:
    """Verify the exact subprocess argv produced for different inputs."""

    @patch("learner.llm.subprocess.run")
    def test_argv_starts_with_claude(self, mock_run):
        mock_run.return_value = _make_completed_process("resp")
        ClaudeCliBackend().complete("hello")
        args = mock_run.call_args[0][0]
        assert args[0] == "claude"

    @patch("learner.llm.subprocess.run")
    def test_argv_includes_dash_p_flag(self, mock_run):
        mock_run.return_value = _make_completed_process("resp")
        ClaudeCliBackend().complete("hello")
        args = mock_run.call_args[0][0]
        assert "-p" in args

    @patch("learner.llm.subprocess.run")
    def test_argv_prompt_follows_dash_p(self, mock_run):
        mock_run.return_value = _make_completed_process("resp")
        ClaudeCliBackend().complete("my prompt text")
        args = mock_run.call_args[0][0]
        p_index = args.index("-p")
        assert args[p_index + 1] == "my prompt text"

    @patch("learner.llm.subprocess.run")
    def test_argv_includes_model_flag(self, mock_run):
        mock_run.return_value = _make_completed_process("resp")
        ClaudeCliBackend().complete("hello")
        args = mock_run.call_args[0][0]
        assert "--model" in args

    @patch("learner.llm.subprocess.run")
    def test_argv_model_value_is_default(self, mock_run):
        mock_run.return_value = _make_completed_process("resp")
        ClaudeCliBackend().complete("hello")
        args = mock_run.call_args[0][0]
        m_index = args.index("--model")
        assert args[m_index + 1] == "claude-sonnet-4-6"

    @patch("learner.llm.subprocess.run")
    def test_argv_model_value_reflects_custom_model(self, mock_run):
        mock_run.return_value = _make_completed_process("resp")
        ClaudeCliBackend(model="claude-opus-4-8").complete("hello")
        args = mock_run.call_args[0][0]
        m_index = args.index("--model")
        assert args[m_index + 1] == "claude-opus-4-8"

    @patch("learner.llm.subprocess.run")
    def test_argv_is_a_list(self, mock_run):
        mock_run.return_value = _make_completed_process("resp")
        ClaudeCliBackend().complete("x")
        assert isinstance(mock_run.call_args[0][0], list)


class TestClaudeCliBackendSubprocessFlags:
    """subprocess.run must be called with capture_output, text, and check."""

    @patch("learner.llm.subprocess.run")
    def test_capture_output_true(self, mock_run):
        mock_run.return_value = _make_completed_process("out")
        ClaudeCliBackend().complete("x")
        kwargs = mock_run.call_args[1]
        assert kwargs.get("capture_output") is True

    @patch("learner.llm.subprocess.run")
    def test_text_mode_true(self, mock_run):
        mock_run.return_value = _make_completed_process("out")
        ClaudeCliBackend().complete("x")
        kwargs = mock_run.call_args[1]
        assert kwargs.get("text") is True

    @patch("learner.llm.subprocess.run")
    def test_check_true(self, mock_run):
        mock_run.return_value = _make_completed_process("out")
        ClaudeCliBackend().complete("x")
        kwargs = mock_run.call_args[1]
        assert kwargs.get("check") is True


class TestClaudeCliBackendOutputStripping:
    """stdout must be stripped of leading/trailing whitespace."""

    @patch("learner.llm.subprocess.run")
    def test_plain_output_returned(self, mock_run):
        mock_run.return_value = _make_completed_process("hello")
        assert ClaudeCliBackend().complete("x") == "hello"

    @patch("learner.llm.subprocess.run")
    def test_trailing_newline_stripped(self, mock_run):
        mock_run.return_value = _make_completed_process("hello\n")
        assert ClaudeCliBackend().complete("x") == "hello"

    @patch("learner.llm.subprocess.run")
    def test_leading_whitespace_stripped(self, mock_run):
        mock_run.return_value = _make_completed_process("  hello")
        assert ClaudeCliBackend().complete("x") == "hello"

    @patch("learner.llm.subprocess.run")
    def test_surrounding_whitespace_stripped(self, mock_run):
        mock_run.return_value = _make_completed_process("\n  response  \n")
        assert ClaudeCliBackend().complete("x") == "response"

    @patch("learner.llm.subprocess.run")
    def test_empty_stdout_returns_empty_string(self, mock_run):
        mock_run.return_value = _make_completed_process("")
        assert ClaudeCliBackend().complete("x") == ""

    @patch("learner.llm.subprocess.run")
    def test_multiline_output_interior_newlines_preserved(self, mock_run):
        mock_run.return_value = _make_completed_process("line1\nline2")
        result = ClaudeCliBackend().complete("x")
        assert "line1" in result and "line2" in result

    @patch("learner.llm.subprocess.run")
    def test_prompt_passed_verbatim_to_subprocess(self, mock_run):
        mock_run.return_value = _make_completed_process("ok")
        tricky_prompt = "line1\nline2\ttabbed"
        ClaudeCliBackend().complete(tricky_prompt)
        args = mock_run.call_args[0][0]
        p_index = args.index("-p")
        assert args[p_index + 1] == tricky_prompt


class TestClaudeCliBackendErrorPaths:
    """Error conditions must propagate without being swallowed."""

    @patch("learner.llm.subprocess.run")
    def test_called_process_error_propagates(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["claude", "-p", "x"]
        )
        with pytest.raises(subprocess.CalledProcessError):
            ClaudeCliBackend().complete("x")

    @patch("learner.llm.subprocess.run")
    def test_file_not_found_propagates(self, mock_run):
        mock_run.side_effect = FileNotFoundError("claude not found")
        with pytest.raises(FileNotFoundError):
            ClaudeCliBackend().complete("x")

    @patch("learner.llm.subprocess.run")
    def test_os_error_propagates(self, mock_run):
        mock_run.side_effect = OSError("permission denied")
        with pytest.raises(OSError):
            ClaudeCliBackend().complete("x")

    @patch("learner.llm.subprocess.run")
    def test_called_process_error_returncode_preserved(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=127, cmd=["claude"]
        )
        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            ClaudeCliBackend().complete("x")
        assert exc_info.value.returncode == 127

    @patch("learner.llm.subprocess.run")
    def test_subprocess_called_once_per_complete(self, mock_run):
        mock_run.return_value = _make_completed_process("ok")
        ClaudeCliBackend().complete("prompt")
        assert mock_run.call_count == 1

    @patch("learner.llm.subprocess.run")
    def test_multiple_calls_each_invoke_subprocess(self, mock_run):
        mock_run.return_value = _make_completed_process("ok")
        backend = ClaudeCliBackend()
        backend.complete("p1")
        backend.complete("p2")
        assert mock_run.call_count == 2


# ── get_backend factory ───────────────────────────────────────────────────────

class TestGetBackendMockPath:
    """All inputs that should return a MockLLM."""

    def test_string_mock_returns_mock_llm(self):
        assert isinstance(get_backend("mock"), MockLLM)

    def test_bool_true_returns_mock_llm(self):
        assert isinstance(get_backend(True), MockLLM)

    def test_env_var_set_returns_mock_llm(self, monkeypatch):
        monkeypatch.setenv("LEARNER_MOCK", "1")
        assert isinstance(get_backend(), MockLLM)

    def test_env_var_set_non_empty_returns_mock_llm(self, monkeypatch):
        monkeypatch.setenv("LEARNER_MOCK", "yes")
        assert isinstance(get_backend("auto"), MockLLM)

    def test_env_var_overrides_auto(self, monkeypatch):
        monkeypatch.setenv("LEARNER_MOCK", "true")
        result = get_backend("auto")
        assert isinstance(result, MockLLM)


class TestGetBackendClaudePath:
    """All inputs that should return a ClaudeCliBackend."""

    def test_default_auto_returns_claude_backend(self, monkeypatch):
        monkeypatch.delenv("LEARNER_MOCK", raising=False)
        assert isinstance(get_backend(), ClaudeCliBackend)

    def test_string_auto_returns_claude_backend(self, monkeypatch):
        monkeypatch.delenv("LEARNER_MOCK", raising=False)
        assert isinstance(get_backend("auto"), ClaudeCliBackend)

    def test_string_claude_returns_claude_backend(self, monkeypatch):
        monkeypatch.delenv("LEARNER_MOCK", raising=False)
        assert isinstance(get_backend("claude"), ClaudeCliBackend)

    def test_bool_false_returns_claude_backend(self, monkeypatch):
        # False is not True and not "mock" so the else branch fires
        monkeypatch.delenv("LEARNER_MOCK", raising=False)
        assert isinstance(get_backend(False), ClaudeCliBackend)

    def test_env_var_absent_auto_gives_claude(self, monkeypatch):
        monkeypatch.delenv("LEARNER_MOCK", raising=False)
        assert isinstance(get_backend("auto"), ClaudeCliBackend)


class TestGetBackendReturnTypeContract:
    """Regardless of which backend is chosen, the result must implement LLMBackend."""

    def test_mock_result_is_llm_backend(self):
        assert isinstance(get_backend("mock"), LLMBackend)

    def test_claude_result_is_llm_backend(self, monkeypatch):
        monkeypatch.delenv("LEARNER_MOCK", raising=False)
        assert isinstance(get_backend("auto"), LLMBackend)

    def test_bool_true_result_is_llm_backend(self):
        assert isinstance(get_backend(True), LLMBackend)

    def test_result_has_complete_method(self):
        backend = get_backend("mock")
        assert callable(getattr(backend, "complete", None))


class TestGetBackendEnvVarEdgeCases:
    """Edge cases around LEARNER_MOCK env-var interaction."""

    def test_env_var_empty_string_does_not_trigger_mock(self, monkeypatch):
        # os.environ.get("LEARNER_MOCK") returns "" which is falsy
        monkeypatch.setenv("LEARNER_MOCK", "")
        result = get_backend("auto")
        # empty string is falsy → should give ClaudeCliBackend
        assert isinstance(result, ClaudeCliBackend)

    def test_env_var_cleared_after_monkeypatch(self, monkeypatch):
        monkeypatch.setenv("LEARNER_MOCK", "1")
        assert isinstance(get_backend(), MockLLM)
        # After the test, monkeypatch restores the env — verified by the next test
        # (monkeypatch is function-scoped so isolation is guaranteed by pytest)

    def test_backend_string_mock_ignores_env(self, monkeypatch):
        # Even with no env var, "mock" string still yields MockLLM
        monkeypatch.delenv("LEARNER_MOCK", raising=False)
        assert isinstance(get_backend("mock"), MockLLM)
