# api-morshu

A FastAPI service that synthesizes Morshu's voice as speech.
It converts text to a WAV file, or to a lip-synced MP4 video, using numpy, the CMU Pronouncing Dictionary, and FFmpeg.
The source recording `morshu.wav` ships inside the package, and `TTS_SOURCE_WAV` can point to a replacement.
Bots and other clients call this API so they do not need to bundle the TTS engine or its dependencies locally.
This project is based on [api-template](https://github.com/Lempki/api-template).
The shared conventions live in [dev-standards](https://github.com/Lempki/dev-standards), and its README is the rulebook for code, prose, commits, and engineering guidelines.
Read it before changing code. When the repositories are cloned side by side, the local copy is `../dev-standards/README.md`.

## Commands

* `uv sync` installs the package and its locked dependencies into `.venv`.
* `uv run uvicorn tts_api.main:app --reload` starts the API. It reads its settings from `.env`.
* `docker compose up --build` builds and runs the service. It listens on host port 8002.
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

* `morshutalk/audio.py` replaces pydub, and `morshutalk/g2p.py` carries g2p-en's model without NLTK. Do not add pydub, g2p-en, or NLTK back.
* `audio.py` cuts at `int(ms * (rate / 1000.0))`, exactly as pydub did, so every clip starts on the same sample as before.
* The engine reads only files inside the package, so it never downloads anything.
* `morshutalk/phrases.py` plays word runs that morshu.wav says verbatim as recorded, using `morshu_words.tsv`. A run needs `MIN_WORDS` words and `MIN_CHARS` characters.
* Change `morshu_words.tsv`, the phoneme table in `morshu.py`, and `morshu.wav` together. `tests/test_phrases.py` checks that they line up.
* Text without a recorded run goes through the phoneme engine exactly as it did before phrase matching.
* Every file in `g2p_data/` stays under 1 MB, the pre-commit limit. The model is therefore one `.npy` file per weight matrix.
* `morshu.wav` is package data. Hatchling puts it in the wheel, so the image needs no volume.
* `/tts/synthesize` answers with a plain `Response`. A failed or timed-out ffmpeg run becomes a 500 with a fixed detail, and its stderr goes only to the log.
* `/tts/phonemes` returns only the phoneme list, never a server path.

## Template rules

* `src/tts_api/auth.py`, `src/tts_api/logging_config.py`, `src/tts_api/service.py`, `tests/test_shared.py`, `.dockerignore`, `setup.sh`, `setup.bat`, `.pre-commit-config.yaml`, and `.github/dependabot.yml` are kept identical to api-template.
* Run `uv run --project ../dev-standards dev-standards template-check --template ../api-template` to check for drift from those files.
* Keep the version only in pyproject.toml, and keep `SERVICE` in main.py equal to the project name there.
* `uv run mypy src` must pass in strict mode, because CI runs it.
