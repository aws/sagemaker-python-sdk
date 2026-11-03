# Multi-stage build to fit AgentCore Runtime's hard 2048 MB image cap.
# Stage 1 installs deps (needs git for the amazon-sop-bench git pin); stage 2 is a slim runtime
# that copies only the installed packages + app code, dropping git, apt lists, pip, and caches.
# All deps resolve from public PyPI + the public SOP-Bench git repo.

# ---- Stage 1: builder ----
# No hardcoded --platform: the build host sets the arch (arm64 natively in cloud CodeBuild for the
# AgentCore Runtime; amd64 natively for `agentcore deploy --local` on an amd64 dev box). Hardcoding
# --platform=linux/arm64 breaks local builds with "exec format error" on amd64 hosts.
FROM public.ecr.aws/docker/library/python:3.11-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install runtime deps and the package into an isolated prefix we can copy wholesale.
COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
COPY pyproject.toml ./
COPY src/ ./src/
# --no-deps: the dependencies were installed by the previous layer.
RUN pip install --no-cache-dir --no-deps --prefix=/install .

# Prune to shrink the image: bytecode caches, test suites, and packaging metadata docs.
# (Kept: .dist-info top-level so imports relying on metadata still work; only bulky bits removed.)
RUN set -eux; \
    find /install -type d -name '__pycache__' -prune -exec rm -rf {} + ; \
    find /install -type d -name 'tests' -prune -exec rm -rf {} + ; \
    find /install -type d -name 'test' -prune -exec rm -rf {} + ; \
    find /install -name '*.pyc' -delete ; \
    find /install -name '*.pyo' -delete ; \
    find /install -name '*.c' -delete ; \
    find /install -name '*.h' -delete ; \
    find /install -name '*.pyi' -delete ; \
    python -m compileall -q --invalidation-mode unchecked-hash /install || true
# Bytecode is precompiled above so cold starts do not recompile every import.

# ---- Stage 2: slim runtime ----
FROM public.ecr.aws/docker/library/python:3.11-slim

WORKDIR /app

# Copy the installed packages + console scripts from the builder.
COPY --from=builder /install /usr/local
# App code (the AgentCore entrypoint references src/sopbench_agentcore/main.py).
COPY src/ ./src/

# Defaults; overridden via `agentcore deploy --env` or .bedrock_agentcore.yaml.
ENV BENCHMARK_NAME=aircraft_inspection \
    MAX_TURNS=10 \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1

# Run as an unprivileged user (port 8080 needs no root).
RUN useradd --create-home --uid 1000 agent
USER agent

EXPOSE 8080

# BedrockAgentCoreApp serves GET /ping on 8080.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/ping', timeout=4)"]

CMD ["python", "-m", "sopbench_agentcore.main"]
