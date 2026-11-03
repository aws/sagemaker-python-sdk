"""Offline tests for the rollout agent: no deployment, endpoint, or AWS calls.

Run:  PYTHONPATH=src python3 scripts/test_local.py

1. Agent loop (no extra dependencies): a scripted fake client drives run_openai_agent.
2. Reward and rollout entrypoint: need amazon_sop_bench (and bedrock_agentcore / sagemaker-train
   for the entrypoint); each group reports SKIP if its dependencies are not installed.
Exits non-zero if any test fails.
"""

import contextlib
import json
import os
import sys

FAILED = []


# --------------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------------
class _Fn:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _ToolCall:
    def __init__(self, id, name, arguments):
        self.id, self.type, self.function = id, "function", _Fn(name, arguments)


class _Message:
    def __init__(self, content="", tool_calls=None, reasoning=None):
        self.content, self.tool_calls, self.reasoning_content = content, tool_calls, reasoning


class _Choice:
    def __init__(self, message):
        self.message = message
        self.finish_reason = "tool_calls" if message.tool_calls else "stop"


class _Response:
    def __init__(self, message):
        self.choices, self.id, self.usage = [_Choice(message)], "fake", None


class _APIError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class ScriptedClient:
    """Returns the scripted messages in order; an int in the script raises that HTTP status."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0
        self.chat = self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, int):
            raise _APIError(item)
        return _Response(item)


TOOLS = [
    {
        "type": "function",
        "function": {"name": "lookup", "description": "", "parameters": {"type": "object"}},
    }
]
ANSWER = '<task_result><answer_json>{"status": "pass"}</answer_json></task_result>'


def _run_agent(script, executor=None, max_turns=5):
    from sopbench_agentcore.openai_agent import run_openai_agent

    calls = []

    def default_executor(name, args):
        calls.append((name, args))
        return '{"record": "found"}'

    client = ScriptedClient(script)
    final, messages = run_openai_agent(
        {"aircraft": "N1"},
        "Inspect the aircraft.",
        TOOLS,
        executor or default_executor,
        client,
        "test-model",
        max_turns=max_turns,
    )
    return final, messages, calls, client


@contextlib.contextmanager
def _env(**values):
    """Set env vars for the block, then restore the previous values."""
    old = {k: os.environ.get(k) for k in values}
    os.environ.update({k: v for k, v in values.items() if v is not None})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except Exception as e:  # noqa: BLE001 - report and keep going
        FAILED.append(name)
        print(f"FAIL  {name}: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------------------------
# 1. Agent loop
# --------------------------------------------------------------------------------------------
def agent_tests():
    tool_turn = _Message("checking", [_ToolCall("c1", "lookup", '{"id": "N1"}')])

    def loop():
        final, messages, calls, _ = _run_agent([tool_turn, _Message(ANSWER)])
        assert calls == [("lookup", {"id": "N1"})], calls
        assert final == ANSWER, final
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant"]

    def reasoning_carried_but_not_scored_on_tool_turns():
        thinking = "thinking: the example shows " + ANSWER
        leak = _Message("", [_ToolCall("c1", "lookup", "{}")], reasoning=thinking)
        final, messages, _, _ = _run_agent([leak, _Message("", None, reasoning=ANSWER)])
        assert messages[2]["content"] == thinking, "the model must see its own reasoning"
        assert final == ANSWER, "tool-free turn reasoning is the answer"

    def bad_tool_json_fed_back():
        bad = _Message("", [_ToolCall("c1", "lookup", '{"id": "N1"')])
        _, messages, calls, _ = _run_agent([bad, _Message(ANSWER)])
        assert calls == [], "tool must not run with invalid arguments"
        assert "not a valid JSON object" in messages[3]["content"], messages[3]

    def api_error_fails_rollout():
        tool_turn = _Message("checking", [_ToolCall("c1", "lookup", "{}")])
        for status in (400, 500):
            try:
                _run_agent([tool_turn, status])
            except _APIError:
                continue
            raise AssertionError(f"a {status} must fail the rollout (so it is retried)")

    def empty_responses_resampled_then_fail():
        from sopbench_agentcore.openai_agent import (
            EMPTY_RESPONSE_MAX_RETRIES,
            EmptyInferenceResponse,
        )

        final, _, _, client = _run_agent([_Message(""), _Message(ANSWER)])
        assert final == ANSWER and client.calls == 2
        try:
            _run_agent([_Message("")] * (EMPTY_RESPONSE_MAX_RETRIES + 1))
        except EmptyInferenceResponse:
            return
        raise AssertionError("persistent empty responses must raise EmptyInferenceResponse")

    def harmony_split_keeps_leading_answer():
        from sopbench_agentcore.openai_agent import _split_harmony_channels

        content, _ = _split_harmony_channels(ANSWER + "<|channel|>commentary")
        assert ANSWER in content, content
        content, _ = _split_harmony_channels("assistant<|channel|>final<|message|>" + ANSWER)
        assert content == ANSWER, content  # a bare role prefix is header noise
        content, _ = _split_harmony_channels("assistant<|channel|>commentary to=functions.x")
        assert content == "", content  # header-only: empty, so it is resampled

    for name, fn in [
        ("agent loop: tool call then answer", loop),
        (
            "agent loop: reasoning is carried in the transcript, scored only on tool-free turns",
            reasoning_carried_but_not_scored_on_tool_turns,
        ),
        ("agent loop: invalid tool JSON is fed back, tool not run", bad_tool_json_fed_back),
        ("agent loop: endpoint errors (400/500) fail the rollout", api_error_fails_rollout),
        ("agent loop: empty responses resampled, then fail", empty_responses_resampled_then_fail),
        (
            "agent loop: Harmony split keeps content, drops header-only noise",
            harmony_split_keeps_leading_answer,
        ),
    ]:
        check(name, fn)


# --------------------------------------------------------------------------------------------
# 2. Reward (needs amazon_sop_bench)
# --------------------------------------------------------------------------------------------
def reward_tests():
    try:
        from sopbench_agentcore.reward import FINAL_REWARD_SCALE, NO_ANSWER_REWARD, compute_reward
        from sopbench_agentcore.samples import load_instance
    except ImportError as e:
        print(f"SKIP  reward tests (amazon_sop_bench not importable: {e})")
        return

    gt = load_instance()["ground_truth"]  # aircraft_inspection_110: 6 fields, 2 not applicable
    nan = [k for k, v in gt.items() if isinstance(v, float) and v != v]
    full = {k: ("NA" if k in nan else v) for k, v in gt.items()}

    def tag(d):
        return "<task_result><answer_json>" + json.dumps(d) + "</answer_json></task_result>"

    def score(answer, messages=None):
        msgs = messages if messages is not None else [{"role": "assistant", "content": answer}]
        return compute_reward(msgs, gt)

    def contract_correct():
        r = score(tag(full))
        assert (r["reward"], r["dense"], r["binary"]) == (FINAL_REWARD_SCALE, 1.0, 1.0), r

    def omitted_na_fields():
        r = score(tag({k: v for k, v in full.items() if k not in nan}))
        assert r["binary"] == 0.0 and abs(r["dense"] - 4 / 6) < 1e-9, r

    def null_counts_as_na():
        r = score(tag({k: (None if k in nan else v) for k, v in full.items()}))
        assert r["binary"] == 1.0, r

    def no_wrapper_is_no_answer():
        r = score(json.dumps(full))
        assert r["reward"] == NO_ANSWER_REWARD and r["answered"] is False, r

    def repeated_answer_credited_once():
        msgs = [{"role": "assistant", "content": tag(full)}] * 2
        assert score(tag(full), msgs)["reward"] == FINAL_REWARD_SCALE

    def last_answer_wins():
        wrong = {k: ("NA" if k in nan else "wrong") for k in gt}
        msgs = [
            {"role": "assistant", "content": tag(full)},
            {"role": "assistant", "content": tag(wrong)},
        ]
        assert score(tag(wrong), msgs)["binary"] == 0.0

    def tool_call_turn_not_scored():
        call = [{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]
        msgs = [{"role": "assistant", "content": tag(full), "tool_calls": call}]
        assert score(None, msgs)["reward"] == NO_ANSWER_REWARD

    def hedged_value_not_credited():
        field = next(k for k in gt if k not in nan and isinstance(gt[k], str))
        hedged = dict(full, **{field: "success or failed or retest"})
        r = score(tag(hedged))
        assert r["binary"] == 0.0 and abs(r["dense"] - 5 / 6) < 1e-9, r

    def partial_single_field_credited():
        field = next(k for k in gt if k not in nan)
        r = score(tag({field: full[field]}))
        assert abs(r["dense"] - 1 / 6) < 1e-9, r

    def universal_hedge_does_not_pay():
        for value, ready in [
            ("success fail retest failed", "true false"),
            ("success-fail-retest", "e"),
            ("success (or fail)", "true/false"),
            ("successfailretest", "truefalse"),
        ]:
            hack = {
                k: ("NA" if k in nan else ready if k == "aircraft_ready" else value) for k in gt
            }
            r = score(tag(hack))
            # At most the not-applicable fields plus a coincidental single-value match: far below
            # the 5.0 a correct answer earns, and never a task success.
            assert r["binary"] == 0.0 and r["reward"] <= FINAL_REWARD_SCALE / 2, (value, r)

    def verbose_single_value_credited():
        verbose = {
            k: (v + " (all checks passed)" if isinstance(v, str) and k not in nan else v)
            for k, v in full.items()
        }
        assert score(tag(verbose))["binary"] == 1.0

    def sop_final_response_format():
        body = ", ".join(
            f"{k!r}: {None if k in nan else (str(v).upper() if isinstance(v, bool) else v)!r}"
            for k, v in gt.items()
        )
        r = score("<final_response>{" + body + "}</final_response>")
        assert (r["reward"], r["binary"]) == (FINAL_REWARD_SCALE, 1.0), r

    def tags_case_insensitive():
        body = json.dumps(full)
        r = score("<TASK_RESULT><ANSWER_JSON>" + body + "</ANSWER_JSON></TASK_RESULT>")
        assert r["binary"] == 1.0, r

    for name, fn in [
        ("reward: contract-correct answer scores 5 / dense 1 / TSR 1", contract_correct),
        ("reward: omitting not-applicable fields loses credit and TSR", omitted_na_fields),
        ("reward: JSON null counts as not applicable", null_counts_as_na),
        ("reward: an untagged answer is no answer (-1)", no_wrapper_is_no_answer),
        ("reward: a repeated answer is credited once", repeated_answer_credited_once),
        ("reward: the last answer is the one scored", last_answer_wins),
        ("reward: an answer on a tool-call turn is not scored", tool_call_turn_not_scored),
        ("reward: a hedged multi-value answer gets no credit", hedged_value_not_credited),
        ("reward: a one-field answer keeps partial credit", partial_single_field_credited),
        ("reward: listing every option is not a universal hack", universal_hedge_does_not_pay),
        ("reward: a single value with an explanation keeps credit", verbose_single_value_credited),
        ("reward: the SOP's own <final_response> format is scored", sop_final_response_format),
        ("reward: answer tags are matched case-insensitively", tags_case_insensitive),
    ]:
        check(name, fn)


# --------------------------------------------------------------------------------------------
# 3. Rollout entrypoint (needs amazon_sop_bench, bedrock_agentcore, sagemaker-train)
# --------------------------------------------------------------------------------------------
def entrypoint_tests():
    try:
        from sopbench_agentcore import main
        from sopbench_agentcore.config import AgentConfig
        from sopbench_agentcore.openai_agent import EmptyInferenceResponse
        from sopbench_agentcore.samples import load_instance
    except ImportError as e:
        print(f"SKIP  entrypoint tests (dependency not importable: {e})")
        return

    payload = {"prompt": json.dumps(load_instance())}

    def with_agent(fn):
        def run():
            with _env(LOCAL_TEST="1"):
                original = main._run_openai
                main._run_openai = fn
                try:
                    return main.run_rollout(payload)
                finally:
                    main._run_openai = original

        return run

    def failures_are_errors():
        def empty(*a, **k):
            raise EmptyInferenceResponse("empty")

        def boom(*a, **k):
            raise ValueError("boom")

        for agent in (empty, boom):
            out = with_agent(agent)()
            assert out["status"] == "error" and "reward" not in out, out

    def instance_payload_shape():
        content = "<task_result><answer_json>{}</answer_json></task_result>"
        agent = lambda *a, **k: (content, [{"role": "assistant", "content": content}])  # noqa: E731
        with _env(LOCAL_TEST="1"):
            original, main._run_openai = main._run_openai, agent
            try:
                out = main.run_rollout({"instance": {"prompt": payload["prompt"]}})
            finally:
                main._run_openai = original
        assert out["status"] == "success", out

    def malformed_payloads_are_errors():
        for bad in ({}, {"prompt": "not json"}, {"prompt": json.dumps({"task_input": {"a": 1}})}):
            with _env(LOCAL_TEST="1"):
                out = main.run_rollout(bad)
            assert out["status"] == "error", (bad, out)

    def scored_rollout():
        answer = load_instance()["ground_truth"]
        answer = {k: ("NA" if isinstance(v, float) else v) for k, v in answer.items()}
        content = "<task_result><answer_json>" + json.dumps(answer) + "</answer_json></task_result>"
        out = with_agent(lambda *a, **k: (content, [{"role": "assistant", "content": content}]))()
        assert out["status"] == "success" and out["reward"] == 5.0, out

    def snake_case_params():
        cfg = AgentConfig()
        p = main._inference_params({"inference_params": {"max_tokens": 77, "top_p": 0.5}}, cfg)
        assert (p["max_tokens"], p["top_p"]) == (77, 0.5), p
        p = main._inference_params({"inferenceParams": {"maxTokens": 88, "topP": 0.4}}, cfg)
        assert (p["max_tokens"], p["top_p"]) == (88, 0.4), p

    def config_validation():
        for bad in ({"BENCHMARK_NAME": "content_flagging"}, {"MAX_TURNS": "0"}):
            with _env(**bad):
                try:
                    AgentConfig.from_env()
                except ValueError:
                    continue
            raise AssertionError(f"{bad} must be rejected")

    for name, fn in [
        (
            "entrypoint: agent failures are reported as failed rollouts, not scored",
            failures_are_errors,
        ),
        ("entrypoint: malformed payloads are reported as errors", malformed_payloads_are_errors),
        ("entrypoint: the SDK RolloutRequest instance shape is accepted", instance_payload_shape),
        ("entrypoint: a correct answer is scored 5", scored_rollout),
        ("entrypoint: sampling params accepted in camelCase and snake_case", snake_case_params),
        ("entrypoint: unsupported config is rejected", config_validation),
    ]:
        check(name, fn)


if __name__ == "__main__":
    agent_tests()
    reward_tests()
    entrypoint_tests()
    print(f"=== {'FAILED: ' + ', '.join(FAILED) if FAILED else 'all passed'} ===")
    sys.exit(1 if FAILED else 0)
