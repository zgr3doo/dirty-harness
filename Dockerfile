FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# setup dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    default-jre-headless \
    graphviz \
    curl \
    && rm -rf /var/lib/apt/lists/*

# PlantUML
ARG PLANTUML_VERSION=1.2026.5
RUN curl -fsSL \
    "https://github.com/plantuml/plantuml/releases/download/v${PLANTUML_VERSION}/plantuml-${PLANTUML_VERSION}.jar" \
    -o /opt/plantuml.jar

ENV PLANTUML_JAR=/opt/plantuml.jar

WORKDIR /app

# Najpierw zależności, aby wykorzystać cache Dockera
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

COPY run.sh .
RUN chmod +x /app/run.sh

RUN mkdir -p /app

RUN mkdir -p /workspace

CMD ["sh", "-c", "/app/run.sh"]