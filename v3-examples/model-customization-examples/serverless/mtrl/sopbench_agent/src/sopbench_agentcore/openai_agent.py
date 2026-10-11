"""Multi-turn tool-use agent over an OpenAI-compatible client (the SageMaker AI Job Runtime).

The loop: send system (SOP) + user (task), let the model call tools, feed the results back, and
repeat until the model answers without calling a tool or max_turns is reached.

Returns (final_output, messages). `messages` is the OpenAI-format transcript; reward.py scores
the last assistant turn that contains a tagged answer.
"""

import json
import logging
from typing import Callable

from .prompts import build_system_prompt

logger = logging.getLogger(__name__)


class EmptyInferenceResponse(RuntimeError):
    """The endpoint kept returning an empty HTTP 200 (no content, reasoning, or tool calls) after
    EMPTY_RESPONSE_MAX_RETRIES in-loop resamples. run_rollout reports the rollout as failed, and
    the trainer retries it up to `rollout_max_retries`."""


# --- Uniform response wrappers so the agent loop is identical for streaming vs non-streaming ---
class _ToolCallFn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.type = "function"
        self.function = _ToolCallFn(name, arguments)


class _Msg:
    def __init__(self, content, reasoning_content, tool_calls):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls or None


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Resp:
    """Duck-typed stand-in for a non-streaming ChatCompletion: exposes .choices[0].message,
    .choices[0].finish_reason, .id, .usage — the exact surface the agent loop reads."""

    class _Choice:
        def __init__(self, message, finish_reason):
            self.message = message
            self.finish_reason = finish_reason

    def __init__(self, message, finish_reason, resp_id, usage):
        self.choices = [self._Choice(message, finish_reason)]
        self.id = resp_id
        self.usage = usage


# Harmony control markers (gpt-oss). Format: <|start|>role<|channel|>chan<|message|>body<|end|>.
# In streaming these can arrive raw in deltas; they must never reach message role/content.
_HARMONY_MARKERS = (
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|channel|>",
    "<|return|>",
    "<|call|>",
    "<|constrain|>",
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
)


# A bare role name before the first channel header is Harmony header noise, not content.
_HARMONY_ROLES = {"assistant", "user", "system", "developer", "tool"}


def _strip_harmony(text):
    """Truncate text at the first Harmony control marker and drop any stray markers, so no
    `<|...|>` token or its trailing channel/role annotation survives into role/content.
    `assistant<|channel|>final_output` -> `assistant`. Used as the fallback when content has no
    parseable <|message|> body (the channel-aware split handles the structured case)."""
    if not text:
        return text or ""
    import re

    # Cut at the first control marker: everything after it is Harmony header noise, not body.
    cut = re.split(r"<\|[a-z_]+\|>", text, maxsplit=1)[0]
    return cut.strip()


def _clean_tool_name(name):
    """Strip Harmony channel annotations that can bleed into tool NAMES.

    Two forms observed in gpt-oss streaming responses:
      - "VerifyAircraftClearance<|channel|>commentary"   (raw Harmony control marker)
      - "VerifyElectricalSystems%commentary"             (marker decoded to '%'; seen in eng logs)

    Tool names in this codebase are ASCII identifiers ([A-Za-z0-9_]) so truncate at the first
    non-identifier character. Handles both known forms without needing a marker-form allowlist.
    Returns "" for a name that starts with a non-identifier char (caller falls back to error path).
    """
    if not name:
        return name or ""
    import re

    m = re.match(r"[A-Za-z0-9_]+", name)
    return m.group(0) if m else ""


def _resolve_tool_name(raw, valid_names):
    """Resolve a model-emitted tool name to a real tool, recovering from two corruption modes:

      1. Channel bleed handled by _clean_tool_name (`Verify...<|channel|>commentary` /
      `...%commentary`).
      2. Hallucinated SUFFIX on an otherwise-valid identifier, e.g. `ReportComponentMismatchjson`
         (the model concatenated `json`) or `VerifyElectricalSystemstool`. _clean_tool_name keeps
         these intact because the suffix is valid identifier text, so the executor 404s and the
         trajectory can fail. Recover by matching against the KNOWN tool list.

    Resolution order:
      - cleaned == a valid name                       -> use it (fast path, no change)
      - cleaned starts with exactly one valid name    -> that name (strip the hallucinated suffix)
      - a valid name starts with cleaned (truncation) -> that name
      - otherwise                                     -> cleaned (caller surfaces "not found")

    Returns (resolved_name, changed_bool). `changed` is True only when we altered the cleaned name,
    so the caller can log the recovery without spamming on the common exact-match path.
    """
    cleaned = _clean_tool_name(raw)
    valid = set(valid_names or ())
    if not cleaned or cleaned in valid:
        return cleaned, False
    # Longest valid name that `cleaned` starts with (handles a trailing hallucinated suffix).
    prefixes = sorted((v for v in valid if cleaned.startswith(v)), key=len, reverse=True)
    if prefixes:
        return prefixes[0], True
    # Valid name that starts with `cleaned` (a truncated name); require a unique hit.
    completions = [v for v in valid if v.startswith(cleaned)]
    if len(completions) == 1:
        return completions[0], True
    return cleaned, False


def _split_harmony_channels(text):
    """If `text` has raw Harmony channel segments, split into (final_content, analysis_reasoning).
    Returns (None, None) when there are no channel markers (normal already-parsed content).

    Harmony body looks like: <|channel|>analysis<|message|>...<|channel|>final<|message|>...
    We route the `analysis` channel to reasoning and everything else (final/commentary) to content.
    """
    if not text or "<|channel|>" not in text:
        return None, None
    import re

    final_parts, analysis_parts = [], []
    segments = text.split("<|channel|>")
    # Text before the first channel header is ordinary content (e.g. an answer followed by a
    # stray trailing header); keep it, minus any other control markers.
    leading = re.split(r"<\|[a-z_]+\|>", segments[0])[0]
    if leading.strip() and leading.strip().lower() not in _HARMONY_ROLES:
        final_parts.append(leading)
    # Each later segment looks like "<chan_name><|message|>body".
    for seg in segments[1:]:
        if "<|message|>" in seg:
            chan, body = seg.split("<|message|>", 1)
            chan = chan.strip().lower()
            # body runs until the next control marker
            body = re.split(r"<\|[a-z_]+\|>", body)[0]
            (analysis_parts if "analysis" in chan else final_parts).append(body)
        # a segment with no <|message|> is header noise -> ignore
    return (
        "".join(final_parts) if final_parts else "",
        "".join(analysis_parts) if analysis_parts else "",
    )


# Delta fields the streaming accumulator KNOWS how to read. Any delta attribute outside this set
# (that carries a value) is an UNSEEN field — a prime suspect when tokens are generated but neither
# content nor reasoning is captured (the server may route the answer to a field we don't drain).
_KNOWN_DELTA_FIELDS = {
    "content",
    "reasoning_content",
    "reasoning",
    "tool_calls",
    "role",
    "function_call",
    "refusal",
}


def _delta_fields(delta):
    """Introspect a streaming delta into (non_empty, empty) field dicts.

    Distinguishing "field present but empty" from "field absent" is essential: a case where tokens
    are generated but content/reasoning stay empty could be either the server sending empty-string
    deltas (field present, empty) or routing text to a field we don't read (field present,
    non-empty, UNSEEN) — the two demand different fixes, so we must tell them apart.

    Also drains pydantic's `model_extra` (fields the SDK model doesn't declare — where a server's
    non-standard field would land) so a routed-elsewhere payload cannot hide.

    Returns (non_empty, empty):
      non_empty: {field: repr-preview}  for attributes carrying a value
      empty:     {field: <type>}        for attributes present but empty ("" / [] / {} / None)
    """
    data = None
    for getter in ("model_dump", "dict"):
        fn = getattr(delta, getter, None)
        if callable(fn):
            try:
                data = fn()
                break
            except Exception:
                data = None
    if data is None:
        data = {
            k: getattr(delta, k, None)
            for k in dir(delta)
            if not k.startswith("_") and not callable(getattr(delta, k, None))
        }
    data = dict(data or {})
    # pydantic v2 stashes undeclared server fields here; surface them explicitly.
    extra = getattr(delta, "model_extra", None)
    if isinstance(extra, dict):
        for k, v in extra.items():
            data.setdefault(k, v)
    non_empty, empty = {}, {}
    for k, v in data.items():
        if v in (None, "", [], {}):
            empty[k] = type(v).__name__
            continue
        s = repr(v)
        non_empty[k] = s if len(s) <= 200 else s[:200] + f"...(+{len(s)-200})"
    return non_empty, empty


def _create_completion(client, stream, verbose=False, tid=None, turn=None, **kwargs):
    """Call chat.completions.create in either mode and return a uniform _Resp-like object.

    Non-streaming: return the SDK object as-is (already has .choices[0].message etc.).
    Streaming: accumulate content / reasoning_content / tool-call deltas across chunks and
    reconstruct a _Resp so the rest of the agent loop is byte-for-byte identical. Streaming is
    gated by the STREAM_MODE env var (see run_openai_agent) so we can toggle without a code change.

    verbose=True (VERBOSE_TRACE env) logs the full request and, for streaming, a per-call summary of
    every raw delta the server sent (field coverage + any UNSEEN field + raw pre-strip previews), so
    a "tokens generated but content/reasoning empty" case is fully diagnosable from CloudWatch
    alone.
    """
    if verbose:
        msgs = kwargs.get("messages") or []
        last = msgs[-1] if msgs else {}
        logger.info(
            "TRACE_REQ tid=%s turn=%s stream=%s model=%s n_messages=%d n_tools=%d "
            "last_role=%s last_content_preview=%r",
            tid,
            turn,
            stream,
            kwargs.get("model"),
            len(msgs),
            len(kwargs.get("tools") or []),
            last.get("role"),
            str(last.get("content"))[:200],
        )

    if not stream:
        resp = client.chat.completions.create(**kwargs)
        if verbose:
            try:
                m = resp.choices[0].message
                u = getattr(resp, "usage", None)
                logger.info(
                    "TRACE_RESP_NONSTREAM tid=%s turn=%s id=%s finish=%s content_len=%d "
                    "reasoning_len=%d n_tool_calls=%d prompt_tokens=%s completion_tokens=%s "
                    "content_preview=%r reasoning_preview=%r",
                    tid,
                    turn,
                    getattr(resp, "id", None),
                    resp.choices[0].finish_reason,
                    len(m.content or ""),
                    len(
                        (
                            getattr(m, "reasoning_content", None)
                            or getattr(m, "reasoning", None)
                            or ""
                        )
                    ),
                    len(m.tool_calls or []),
                    getattr(u, "prompt_tokens", None),
                    getattr(u, "completion_tokens", None),
                    (m.content or "")[:400],
                    str(
                        getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None) or ""
                    )[:400],
                )
            except Exception:
                logger.exception("TRACE_RESP_NONSTREAM introspection failed")
        return resp

    content_parts, reasoning_parts = [], []
    finish_reason, resp_id = None, None
    prompt_tokens, completion_tokens = None, None
    # tool calls arrive as indexed fragments: {index: {id, name, arguments_parts[]}}
    tc_acc = {}
    # --- verbose accounting: per-field chunk coverage + any field we don't read ---
    n_chunks = 0
    n_empty_delta_chunks = 0  # chunks whose delta had NO non-empty field (only empty/absent)
    field_hits = {}  # delta field name -> count of chunks carrying a NON-EMPTY value
    empty_field_hits = {}  # delta field name -> count of chunks where it was present but EMPTY
    unseen_fields = {}  # UNSEEN non-empty field name -> a sample repr (token-loss suspects)
    first_delta_fields = None
    finish_delta_fields = None
    raw_chunk_samples = []  # first few (non_empty, empty) delta shapes, verbatim
    stream_kwargs = dict(kwargs, stream=True)
    if verbose:
        # Token counts on the final chunk, for the diagnostics below (verbose mode only).
        stream_kwargs["stream_options"] = {"include_usage": True}
    for chunk in client.chat.completions.create(**stream_kwargs):
        n_chunks += 1
        if getattr(chunk, "id", None):
            resp_id = chunk.id
        u = getattr(chunk, "usage", None)
        if u is not None:
            prompt_tokens = getattr(u, "prompt_tokens", prompt_tokens)
            completion_tokens = getattr(u, "completion_tokens", completion_tokens)
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if getattr(choice, "finish_reason", None):
            finish_reason = choice.finish_reason
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        if verbose:
            non_empty, empty = _delta_fields(delta)
            if first_delta_fields is None:
                first_delta_fields = non_empty
            if getattr(choice, "finish_reason", None):
                finish_delta_fields = non_empty
            if not non_empty:
                n_empty_delta_chunks += 1
            if len(raw_chunk_samples) < 5:
                raw_chunk_samples.append({"non_empty": non_empty, "empty": empty})
            for k, v in non_empty.items():
                field_hits[k] = field_hits.get(k, 0) + 1
                if k not in _KNOWN_DELTA_FIELDS and k not in unseen_fields:
                    unseen_fields[k] = v  # capture first sample of an unread field
            for k in empty:
                empty_field_hits[k] = empty_field_hits.get(k, 0) + 1
        if getattr(delta, "content", None):
            content_parts.append(delta.content)
        # GPT-OSS reasoning arrives on the same delta under reasoning_content/reasoning
        r = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if r:
            reasoning_parts.append(r)
        for tcd in getattr(delta, "tool_calls", None) or []:
            idx = getattr(tcd, "index", 0)
            slot = tc_acc.setdefault(idx, {"id": None, "name": None, "args": []})
            if getattr(tcd, "id", None):
                slot["id"] = tcd.id
            fn = getattr(tcd, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    slot["name"] = fn.name
                if getattr(fn, "arguments", None):
                    slot["args"].append(fn.arguments)
    tool_calls = [
        _ToolCall(v["id"], _clean_tool_name(v["name"]), "".join(v["args"]))
        for _, v in sorted(tc_acc.items())
        if v["name"]
    ]
    # Harmony-aware reconstruction. Streamed gpt-oss deltas can carry raw Harmony control markers
    # (<|channel|>, <|message|>, <|start|>, <|end|>, <|return|>, <|call|>). If left in, they leak
    # into role/content on the NEXT request and the engine rejects it with
    # "Unknown role: assistant<|channel|>final_output" (openai.APIError), failing the trajectory.
    # The non-streaming SDK path parses these out; the streaming path must do it explicitly.
    raw_content = "".join(content_parts)
    raw_reasoning = "".join(reasoning_parts)
    # If the raw content contains channel headers, route analysis->reasoning, final/other->content.
    chan_content, chan_reasoning = _split_harmony_channels(raw_content)
    content = _strip_harmony(chan_content if chan_content is not None else raw_content)
    reasoning = _strip_harmony((raw_reasoning + (chan_reasoning or "")).strip())
    msg = _Msg(content, reasoning, tool_calls)
    usage = _Usage(prompt_tokens, completion_tokens)

    if verbose:
        # Per-call streaming summary: field coverage across chunks, any UNSEEN field (tokens routed
        # to something we don't read), and raw vs post-strip lengths. Comparing completion_tokens
        # with the raw lengths diagnoses "tokens generated but everything empty".
        logger.info(
            "TRACE_STREAM tid=%s turn=%s id=%s finish=%s n_chunks=%d n_empty_delta_chunks=%d "
            "completion_tokens=%s field_hits=%s empty_field_hits=%s unseen_fields=%s "
            "raw_content_len=%d raw_reasoning_len=%d post_content_len=%d post_reasoning_len=%d "
            "n_tool_calls=%d first_delta=%s finish_delta=%s",
            tid,
            turn,
            resp_id,
            finish_reason,
            n_chunks,
            n_empty_delta_chunks,
            completion_tokens,
            field_hits,
            empty_field_hits,
            unseen_fields,
            len(raw_content),
            len(raw_reasoning),
            len(content),
            len(reasoning or ""),
            len(tool_calls),
            first_delta_fields,
            finish_delta_fields,
        )
        # If tokens were generated but we captured nothing, dump the raw text AND the first few raw
        # delta shapes so the loss is fully diagnosable: empty_field_hits shows fields present-but-
        # empty (server sent ""), unseen_fields shows fields carrying data we don't read, and
        # raw_chunk_samples shows the verbatim first deltas. This disambiguates "server sent empty"
        # from "we dropped it" from "routed to an unread field".
        if (
            (completion_tokens or 0) > 0
            and not content.strip()
            and not (reasoning or "").strip()
            and not tool_calls
        ):
            logger.warning(
                "TRACE_STREAM_EMPTY tid=%s turn=%s id=%s completion_tokens=%s n_chunks=%d "
                "n_empty_delta_chunks=%d empty_field_hits=%s unseen_fields=%s -- tokens generated "
                "but nothing captured. raw_content=%r raw_reasoning=%r raw_chunk_samples=%s",
                tid,
                turn,
                resp_id,
                completion_tokens,
                n_chunks,
                n_empty_delta_chunks,
                empty_field_hits,
                unseen_fields,
                raw_content[:1000],
                raw_reasoning[:1000],
                raw_chunk_samples,
            )

    return _Resp(msg, finish_reason, resp_id, usage)


# How many times to re-issue the SAME sampling call when the engine returns a degenerate empty
# HTTP 200 (no content, no reasoning, no tool calls). An empty 200 is a success to the OpenAI
# client, so client.max_retries never covers it. Resampling in-loop keeps the trajectory alive;
# giving up fails the whole rollout (see EmptyInferenceResponse). gpt-oss emits these on ~1% of
# calls (more on the answer turn), and a few resamples almost always recover.
EMPTY_RESPONSE_MAX_RETRIES = 3


def _is_empty_response(msg, content, reasoning):
    """A degenerate 'empty 200': no content, no reasoning, and no tool calls."""
    return not (content or "").strip() and not msg.tool_calls and not (reasoning or "").strip()


def run_openai_agent(
    task_input,
    sop_text: str,
    tools: list,
    tool_executor: Callable[[str, dict], str],
    client,
    model: str,
    max_turns: int = 10,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    top_p: float = 1.0,
    debug: bool = False,
    trajectory_id=None,
    stream: bool = False,
    verbose_trace: bool = False,
) -> tuple[str, list]:
    user_content = task_input if isinstance(task_input, str) else json.dumps(task_input)
    messages = [
        {"role": "system", "content": build_system_prompt(sop_text)},
        {"role": "user", "content": user_content},
    ]
    # Known tool names, for _resolve_tool_name suffix-recovery (e.g. "ReportComponentMismatchjson").
    valid_tool_names = {
        (t.get("function") or {}).get("name")
        for t in (tools or [])
        if (t.get("function") or {}).get("name")
    }

    final_output = ""
    if debug:
        print(
            f"  [request params] model={model!r} temperature={temperature} "
            f"max_tokens={max_tokens} top_p={top_p} n_tools={len(tools or [])} "
            f"max_turns={max_turns} stream={stream} verbose_trace={verbose_trace}"
        )
    for _turn in range(max_turns):
        # Resample an empty 200 up to EMPTY_RESPONSE_MAX_RETRIES times before giving up.
        response = msg = content = reasoning = finish_reason = None
        for _attempt in range(EMPTY_RESPONSE_MAX_RETRIES + 1):
            try:
                # Gated streaming: when stream=True, _create_completion accumulates chunk deltas and
                # returns a uniform object with the same .choices[0].message / .finish_reason /
                # .usage surface, so everything below is identical to the non-streaming path.
                response = _create_completion(
                    client,
                    stream,
                    verbose=verbose_trace,
                    tid=trajectory_id,
                    turn=_turn,
                    model=model,
                    messages=messages,
                    tools=tools or None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except Exception as e:  # noqa: BLE001 - log, then fail the rollout
                # Endpoint errors (400 context overflow, 500, timeouts, throttling) fail the
                # rollout; the trainer retries it up to `rollout_max_retries`.
                logger.warning(
                    "turn=%s trajectory_id=%s sampling call failed: %s", _turn, trajectory_id, e
                )
                raise
            msg = response.choices[0].message
            content = msg.content or ""
            finish_reason = response.choices[0].finish_reason
            reasoning = (
                getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
            )

            # Always-on per-turn log, greppable by trajectory_id under concurrent rollouts.
            usage = getattr(response, "usage", None)
            logger.info(
                "turn=%s trajectory_id=%s attempt=%s finish_reason=%s content_len=%d "
                "reasoning_len=%d "
                "n_tool_calls=%d response_id=%s prompt_tokens=%s completion_tokens=%s",
                _turn,
                trajectory_id,
                _attempt,
                finish_reason,
                len(content),
                len(reasoning or ""),
                len(msg.tool_calls or []),
                getattr(response, "id", None),
                getattr(usage, "prompt_tokens", None),
                getattr(usage, "completion_tokens", None),
            )

            if not _is_empty_response(msg, content, reasoning):
                break  # got a usable response

            # Empty 200: resample (same messages) unless we've exhausted retries.
            if _attempt < EMPTY_RESPONSE_MAX_RETRIES:
                logger.warning(
                    "turn=%s trajectory_id=%s EMPTY_RESPONSE attempt=%s finish_reason=%s "
                    "response_id=%s "
                    "-- resampling in-loop (before any /complete-rollout)",
                    _turn,
                    trajectory_id,
                    _attempt,
                    finish_reason,
                    getattr(response, "id", None),
                )

        # Still empty after all resamples: fail the rollout (run_rollout reports it as failed)
        # rather than treating the empty turn as a final answer.
        if _is_empty_response(msg, content, reasoning):
            logger.warning(
                "turn=%s trajectory_id=%s EMPTY_RESPONSE_EXHAUSTED after %s resamples "
                "finish_reason=%s "
                "response_id=%s -- raising",
                _turn,
                trajectory_id,
                EMPTY_RESPONSE_MAX_RETRIES,
                finish_reason,
                getattr(response, "id", None),
            )
            raise EmptyInferenceResponse(
                f"empty inference response (trajectory={trajectory_id}, turn={_turn}, "
                f"finish_reason={finish_reason}, after {EMPTY_RESPONSE_MAX_RETRIES} resamples)"
            )

        # gpt-oss often leaves `content` empty and puts its text in the reasoning (analysis)
        # channel. Carry the reasoning as the turn's content so the model keeps its chain of thought
        # across the tool loop (without it, gpt-oss tends to return empty responses mid-task) and
        # so a final answer written only in reasoning is not lost. The reward only reads answers
        # from turns WITHOUT tool calls, so reasoning on a tool-call turn is never scored.
        if not content.strip() and reasoning:
            content = reasoning

        if debug:
            print(f"  [turn {_turn}] finish_reason={finish_reason}")
            print(f"  [turn {_turn}] content(len={len(content)}) tail={content[-600:]!r}")
            if reasoning:
                rc = str(reasoning)
                print(f"  [turn {_turn}] reasoning_content(len={len(rc)}) tail={rc[-600:]!r}")
            print(
                f"  [turn {_turn}] tool_calls="
                f"{[(tc.function.name, tc.function.arguments) for tc in (msg.tool_calls or [])]}"
            )

        # Resolve each model-emitted tool name once: strips Harmony channel bleed
        # ("...<|channel|>commentary") and recovers a hallucinated suffix
        # ("ReportComponentMismatchjson" -> "ReportComponentMismatch") against the known tools.
        resolved = [
            _resolve_tool_name(tc.function.name, valid_tool_names) for tc in (msg.tool_calls or [])
        ]
        assistant_msg = {"role": "assistant", "content": content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": name, "arguments": tc.function.arguments or "{}"},
                }
                for tc, (name, _) in zip(msg.tool_calls, resolved)
            ]
        messages.append(assistant_msg)
        if verbose_trace:
            logger.info(
                "TRACE_ASSISTANT tid=%s turn=%s content=%r tool_calls=%s",
                trajectory_id,
                _turn,
                content[:1000],
                [
                    (tc["function"]["name"], tc["function"]["arguments"][:300])
                    for tc in assistant_msg.get("tool_calls", [])
                ],
            )

        if not msg.tool_calls:
            final_output = content
            break

        for tc, (tool_name, _name_recovered) in zip(msg.tool_calls, resolved):
            if _name_recovered:
                logger.warning(
                    "turn=%s trajectory_id=%s TOOL_NAME_RECOVERED raw=%r -> %r",
                    _turn,
                    trajectory_id,
                    tc.function.name,
                    tool_name,
                )
            # A bad tool call (invalid JSON arguments, unknown name, tool error) is recoverable:
            # feed the error back as the tool result so the model can correct itself next turn.
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                if not isinstance(args, dict):
                    raise json.JSONDecodeError("arguments must be a JSON object", "", 0)
            except json.JSONDecodeError as e:
                args, result = None, f"Tool call arguments are not a valid JSON object: {e}"
            if args is not None:
                try:
                    result = tool_executor(tool_name, args)
                except Exception as e:  # noqa: BLE001 - surface any tool error back to the model
                    result = f"Tool execution failed. Error: {e}"
            if debug and result.startswith(("Tool execution failed", "Tool call arguments")):
                print(f"  [turn {_turn}] tool error fed back: {result[:300]!r}")
            if verbose_trace:
                logger.info(
                    "TRACE_TOOL tid=%s turn=%s name=%r args=%r result=%r",
                    trajectory_id,
                    _turn,
                    tool_name,
                    (tc.function.arguments or "")[:300],
                    str(result)[:600],
                )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    if not final_output:
        # Ended without a tool-free final turn (max_turns, or a 400/500 cut the loop short): use
        # the last assistant content.
        for m in reversed(messages):
            if m.get("role") == "assistant" and m.get("content"):
                final_output = m["content"]
                break

    return final_output, messages
