# discord-api-morshu

A FastAPI service that synthesizes Morshu's voice as speech.
It converts text to a WAV file, or to a lip-synced MP4 video, using g2p-en, pydub, numpy, and FFmpeg.
The source recording `morshu.wav` ships inside the package, and `TTS_SOURCE_WAV` can point to a replacement.
Discord bots call this API so they do not need to bundle the TTS engine or its dependencies locally.
This project is based on [discord-api-template](https://github.com/Lempki/discord-api-template).
The shared conventions live in [discord-dev-standards](https://github.com/Lempki/discord-dev-standards), and its README is the rulebook for code, prose, and commits.

## Commands

* `uv sync` installs the package and its locked dependencies into `.venv`.
* `uv run uvicorn tts_api.main:app --reload` starts the API. It reads its settings from `.env`.
* `docker-compose up --build` builds and runs the service. It listens on host port 8002.
* `uv run pytest` runs the tests.
* `uvx pre-commit run --all-files` runs every lint and format hook.

## Layout

* `src/tts_api/main.py` defines the app, the lifespan, and the routes.
* `src/tts_api/config.py` adds this service's settings to `ServiceSettings`.
* `src/tts_api/service.py` holds `ServiceSettings`, which validates the shared secret, and `service_version()`, which reads the version from pyproject.toml.
* `src/tts_api/logging_config.py` turns every log record, including uvicorn's, into one JSON line.
* `src/tts_api/auth.py` holds the bearer token dependency that protects every route except `/health`.
* `src/tts_api/models.py` holds the request and response models.
* `src/tts_api/morshutalk/` holds the TTS engine adapted from [MorshuTalk](https://github.com/n0spaces/MorshuTalk), including the phoneme matching logic, the grapheme-to-phoneme wrapper, the source recording `morshu.wav`, and the sprite frames used for video synthesis.

## Runtime notes

* The Dockerfile downloads every NLTK package that g2p-en and `init` need at build time, into `/usr/local/share/nltk_data`.
* `init` downloads an NLTK package only when `nltk.data.find` cannot find it, so the container never downloads and local runs still work.
* `morshu.wav` is package data. Hatchling puts it in the wheel, so the image needs no volume.
* `/tts/synthesize` answers with a plain `Response`. A failed or timed-out ffmpeg run becomes a 500 with a fixed detail, and its stderr goes only to the log.
* `/tts/phonemes` returns only the phoneme list, never a server path.

## Template rules

* `src/tts_api/auth.py`, `src/tts_api/logging_config.py`, `src/tts_api/service.py`, `.dockerignore`, `setup.sh`, `setup.bat`, `.pre-commit-config.yaml`, and `.github/dependabot.yml` are kept identical to discord-api-template.
* Run `uv run --project ../discord-dev-standards dev-standards template-check --template ../discord-api-template` to check for drift from those files.
* Keep the version only in pyproject.toml, and keep `SERVICE` in main.py equal to the project name there.
* `uv run mypy src` must pass in strict mode, because CI runs it.
