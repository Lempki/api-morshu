# The uv stage only supplies the uv binary. A named stage lets Dependabot keep its tag current.
FROM ghcr.io/astral-sh/uv:0.12.19 AS uv

FROM python:3.12-slim AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies are installed before the source is copied, so code changes reuse the cached layer.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ src/
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home appuser
WORKDIR /app
# The virtual environment holds the installed package, so the runtime image needs no source tree.
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    NLTK_DATA=/usr/local/share/nltk_data
# The NLTK data is downloaded at build time into a directory that appuser can read.
# Importing g2p-en needs averaged_perceptron_tagger and cmudict.
# The morshutalk init needs averaged_perceptron_tagger_eng and punkt_tab.
# Both fetch only missing packages, so a container start downloads nothing.
# raise_on_error makes a failed download fail the build.
# NLTK saves the zip files readable by root only, so chmod opens them to appuser.
RUN python -c "import nltk, os; [nltk.download(p, download_dir=os.environ['NLTK_DATA'], quiet=True, raise_on_error=True) for p in ('averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'cmudict', 'punkt_tab')]" \
    && chmod -R a+rX "$NLTK_DATA"
USER appuser
EXPOSE 8000
# Docker and compose mark the container unhealthy when /health stops answering.
# Startup loads the G2p model and the source WAV, so the start period is longer than the template's.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["uvicorn", "tts_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
