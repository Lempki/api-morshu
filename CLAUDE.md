# discord-api-morshu

A FastAPI service that synthesizes Morshu's voice as speech.
It converts text to a WAV file, or to a lip-synced MP4 video, using g2p-en, pydub, numpy, and FFmpeg.
It reads the source audio file `morshu.wav` from `/data`, mounted as a read-only volume by docker-compose.
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
* `src/tts_api/config.py` reads settings from the environment with pydantic-settings.
* `src/tts_api/auth.py` holds the bearer token dependency that protects every route except `/health`.
* `src/tts_api/models.py` holds the request and response models.
* `src/tts_api/morshutalk/` holds the TTS engine adapted from [MorshuTalk](https://github.com/n0spaces/MorshuTalk), including the phoneme matching logic, the grapheme-to-phoneme wrapper, and the sprite frames used for video synthesis.

## Template rules

* `src/tts_api/auth.py`, `.dockerignore`, `setup.sh`, `setup.bat`, `.pre-commit-config.yaml`, and `.github/dependabot.yml` are kept identical to discord-api-template.
* Run `uv run --project ../discord-dev-standards dev-standards template-check --template ../discord-api-template` to check for drift from those files.
