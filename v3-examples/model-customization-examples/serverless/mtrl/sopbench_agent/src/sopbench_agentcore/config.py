"""Configuration for the SOP-Bench AgentCore rollout agent.

All fields are environment-driven so the deployed image can be reconfigured at deploy time
(`agentcore deploy --env KEY=VALUE`) without code changes. The SageMaker AI Job Runtime endpoint
is resolved as: RFT_RUNTIME_ENDPOINT -> RFT_RUNTIME_URL -> built from the region.
"""

import os
from dataclasses import dataclass

# This example is built and tested for one multi-field SOP-Bench benchmark.
SUPPORTED_BENCHMARKS = ("aircraft_inspection",)


def _flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


@dataclass
class AgentConfig:
    model_name: str = ""  # model id sent to the endpoint (the training job usually fixes it)
    max_turns: int = 10
    benchmark_name: str = "aircraft_inspection"
    # Sampling fallbacks, used only when the runtime sends no inference params (it normally does).
    # They match the trainer's default hyperparameters: temperature 1.0, sampling_top_p 1.0,
    # sampling_max_tokens 4096.
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 4096
    region: str = ""  # AWS_REGION; AgentCore sets it in the runtime container
    local_test: bool = False  # skip the token path and call LOCAL_ENDPOINT (offline testing)
    debug: bool = False  # verbose per-turn logging
    stream: bool = False  # stream chat/completions (STREAM_MODE)
    verbose_trace: bool = False  # log every request, streaming delta, and response (VERBOSE_TRACE)

    def __post_init__(self):
        if self.benchmark_name not in SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"BENCHMARK_NAME={self.benchmark_name!r} is not supported; "
                f"expected one of {SUPPORTED_BENCHMARKS}"
            )
        if self.max_turns < 1:
            raise ValueError(f"MAX_TURNS must be >= 1, got {self.max_turns}")

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            model_name=os.environ.get("MODEL_NAME", ""),
            max_turns=int(os.environ.get("MAX_TURNS", "10")),
            benchmark_name=os.environ.get("BENCHMARK_NAME", "aircraft_inspection"),
            temperature=float(os.environ.get("TEMPERATURE", "1.0")),
            top_p=float(os.environ.get("TOP_P", "1.0")),
            max_tokens=int(os.environ.get("MAX_TOKENS", "4096")),
            region=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", ""),
            local_test=_flag("LOCAL_TEST"),
            debug=_flag("DEBUG"),
            stream=_flag("STREAM_MODE"),
            verbose_trace=_flag("VERBOSE_TRACE"),
        )

    @property
    def rft_runtime_endpoint(self) -> str:
        """Resolve the SageMaker AI Job Runtime endpoint."""
        env_url = os.environ.get("RFT_RUNTIME_ENDPOINT") or os.environ.get("RFT_RUNTIME_URL")
        if env_url:
            return env_url.rstrip("/")
        if not self.region:
            raise ValueError("AWS_REGION is not set; cannot build the Job Runtime endpoint")
        return f"https://job-runtime.sagemaker.{self.region}.api.aws"
