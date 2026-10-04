# api-morshu

This is a REST API that synthesizes speech in Morshu's voice and returns the result as an audio or video file. It hosts the TTS engine adapted from [MorshuTalk](https://github.com/n0spaces/MorshuTalk) by [n0spaces](https://github.com/n0spaces), converting arbitrary text into audio by stitching phoneme segments from Morshu's original Zelda CD-i dialogue. Clients such as Discord bots call this API to generate and play Morshu audio without bundling the TTS engine or its dependencies locally. This project is based on the [api-template](https://github.com/Lempki/api-template) repository, which provides the core architecture.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/tts/synthesize` | Generate audio or video from text. Returns a WAV or MP4 file depending on the `format` field. |
| `GET` | `/tts/phonemes` | List the phoneme tokens available in the loaded source audio. |
| `GET` | `/health` | Returns the service name and the version set in `pyproject.toml`. Uptime monitors and the Docker health check call it. |

All endpoints except `/health` require a bearer token in the `Authorization` header.
A request without the header or with a wrong token gets `401 Unauthorized` with a `WWW-Authenticate: Bearer` header.
The token is compared in constant time.

### POST /tts/synthesize

```json
{
  "text": "lamp oil, rope, bombs?",
  "speed": 1.0,
  "trim_silence": false,
  "format": "wav",
  "phrase_matching": true
}
```

`speed` accepts values between `0.5` and `2.0`. `trim_silence` removes leading and trailing silence from the output. `format` accepts `"wav"` (default) or `"video"`. `phrase_matching` is on by default and is described below.

When `format` is `"wav"`, returns a binary WAV file with `Content-Type: audio/wav`. When `format` is `"video"`, generates an MP4 by compositing MorshuTalk sprite frames at 10 fps in sync with the synthesised audio and returns the file with `Content-Type: video/mp4`. The `speed` and `trim_silence` fields are ignored for video output. If the text exceeds the configured maximum length or no phoneme matches are found, a `422` response is returned.

The whole file arrives in one response with a `Content-Length` header.
If FFmpeg fails or runs for more than 120 seconds, the video request gets a `500` answer.
Its body is always `{"detail": "Could not encode the video."}`.
The service logs the end of FFmpeg's error output, but the response never includes it.

#### Phrase matching

The phoneme engine builds speech from short clips, so even Morshu's own lines come out choppy when it speaks them.
With `phrase_matching` on, any run of words that morshu.wav says in the same order plays exactly as recorded.
A run needs at least 2 words and 8 characters, so single words and short pairs still go to the phoneme engine.
The search is greedy from left to right and ignores case, accents, apostrophes, and punctuation.
A hum of two or more m's matches the recording's "mmm".
The engine speaks everything else, and the pieces are joined with the engine's usual pauses.

| Text | Result |
|---|---|
| `Lamp oil, rope, bombs, you want it?` | One recorded clip. |
| `Welcome to my shop, my friend!` | The engine says "Welcome to my shop", and "my friend" plays as recorded. The first "my" is a single word, so the engine says it. |
| `Rope, bombs, lamp oil.` | Two recorded clips, "Rope, bombs" and "lamp oil". |
| `As long as you want.` | "As long as you" plays as recorded, and the engine says "want". |

The video follows the same pieces, so recorded runs also get Morshu's original mouth movements.
The word-level transcript of the recording is `morshutalk/morshu_words.tsv`.
A test checks that every word in it lines up with the phoneme table in `morshu.py`.

### GET /tts/phonemes

Returns the phoneme tokens that the source recording contains, as `{"phonemes": ["AA", "AE", ...]}`.
The response holds only that list.

## Prerequisites

* [Docker](https://docs.docker.com/get-docker/) and Docker Compose.

Running without Docker requires Python 3.12, [uv](https://docs.astral.sh/uv/), and FFmpeg available in the system PATH. On Windows, install uv with `winget install --id astral-sh.uv`.

## Assets

The TTS engine cuts every clip from one source recording, `morshu.wav`, which holds Morshu's CD-i dialogue.
It ships inside the package at `src/tts_api/morshutalk/morshu.wav`, next to the sprite frames.
The wheel and the Docker image include it, so the container needs no volume.
To use a different recording, set `TTS_SOURCE_WAV` to its path.

The 154 sprite frames used for video synthesis are adapted from [MorshuTalk](https://github.com/n0spaces/MorshuTalk) and are bundled with the application in `src/tts_api/morshutalk/sprites/`. They do not need to be provided separately.

## Setup

The setup script prepares the project in a single run, and it is safe to run again at any time.

On Windows, double-click `setup.bat` or run it from a terminal:

```
setup.bat
```

On macOS or Linux, run the following commands:

```
chmod +x setup.sh
./setup.sh
```

The script asks before it installs anything, and it does the following:

1. It installs [uv](https://docs.astral.sh/uv/) when uv is missing. uv also provides Python 3.12 when the machine lacks it.
2. It offers to install Docker, and the tools that the Docker image includes for running outside Docker. It uses winget on Windows, Homebrew on macOS, and the system package manager on Linux.
3. It runs `uv sync`, which installs the package and its locked dependencies into `.venv`.
4. It copies `.env.template` to `.env` on the first run and fills `API_SECRET` with a random value.

A step that fails says what went wrong, why it matters, and what to do next, and the summary at the end lists it again.
The steps live in `scripts/bootstrap.py`, which needs only the Python standard library.

If you prefer to perform the setup manually, follow these steps:

```bash
uv sync
cp .env.template .env
# Edit .env and set API_SECRET and other values as needed.
uv run uvicorn tts_api.main:app --port 8002
```

### Docker

Alternatively, you can run the API as a Docker container.

1. Copy `.env.template` to `.env` and set `API_SECRET`.
2. Build and start the container:

   ```
   docker compose up --build
   ```

After starting with Docker Compose, the API is available at `http://localhost:8002`. The container itself listens on port `8000`, and Docker Compose maps host port `8002` to it. The service keeps no state between requests, so the container needs no volume.

The image has a health check that calls `/health` every 30 seconds.
Startup loads the pronunciation dictionary and the grapheme-to-phoneme model, so the check allows 60 seconds before it counts a failure.
`docker ps` shows the container as `healthy` once the API answers.

Every file the engine reads ships inside the package, so the service never downloads anything and needs no network access.

## Configuration

All configuration is read from environment variables or from a `.env` file in the project root.

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_SECRET` | Yes | None | Shared bearer token of at least 16 characters. Every client must send this value in the `Authorization` header. Generate one with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. |
| `TTS_SOURCE_WAV` | No | None | Path to a replacement for the source WAV file, which must be 16-bit PCM in mono or stereo. When it is unset or empty, the service uses the `morshu.wav` that ships in the package. |
| `LOG_LEVEL` | No | `INFO` | Log verbosity. Accepts `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. |
| `TTS_MAX_TEXT_LENGTH` | No | `500` | Maximum number of characters accepted per synthesis request. |

The service refuses to start when `API_SECRET` is shorter than 16 characters or is a placeholder such as `changeme`.
The error names the variable but never repeats its value.

Logs are structured JSON.
Every line is one JSON object, including uvicorn's access log, so log collectors can parse it without guessing.

## Project structure

```
api-morshu/
├── src/tts_api/
│   ├── main.py         # FastAPI application and route definitions.
│   ├── config.py       # This service's settings on top of the shared ones.
│   ├── service.py      # Shared settings, secret validation, and the version lookup.
│   ├── logging_config.py  # JSON log formatter for the app and uvicorn.
│   ├── auth.py         # Bearer token dependency.
│   ├── models.py       # Pydantic request and response models.
│   └── morshutalk/     # TTS engine adapted from MorshuTalk by n0spaces.
│       ├── morshu.py   # Core phoneme matching and audio stitching logic.
│       ├── phrases.py  # Finds word runs that the recording says verbatim.
│       ├── morshu_words.tsv  # Every word of morshu.wav with its start and end time.
│       ├── audio.py    # Mono 16-bit audio clips on numpy and the wave module.
│       ├── g2p.py      # Text to phonemes with the CMU dictionary and a small neural model.
│       ├── numbers.py  # Spells out numbers, money, and ordinals before g2p.
│       ├── g2p_data/   # The CMU dictionary, the model's weights, and their licenses.
│       ├── morshu.wav  # Source recording that every clip is cut from.
│       └── sprites/    # 154 sprite frames for video synthesis (0.png to 153.png).
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml      # Project metadata and dependencies.
├── uv.lock             # Locked dependency versions.
├── ruff.toml           # Lint and format settings on top of the shared baseline.
├── setup.bat           # Windows setup script.
├── setup.sh            # macOS and Linux setup script.
├── scripts/bootstrap.py  # The steps that both setup scripts run.
└── .env.template       # Template for environment variables.
```

## Running tests

```bash
uv run pytest
```

Run every lint and format check with `uvx pre-commit run --all-files`, or install the hooks once with `uvx pre-commit install` so they run on each commit.
The coding, prose, and commit conventions are documented in [dev-standards](https://github.com/Lempki/dev-standards).

## Dependencies

The engine runs on numpy, `inflect` for spelling out numbers, and FFmpeg for video.
It does not use pydub, g2p-en, or NLTK.
pydub and g2p-en are unmaintained, and pydub needs the `audioop` module that Python 3.13 removed.
`audio.py` replaces pydub with numpy and the standard library's `wave` module.
`g2p.py` carries g2p-en's model and code, without its NLTK parts.
Pronunciations come from the CMU Pronouncing Dictionary that ships in `g2p_data/`.
A word with several pronunciations takes the dictionary's first one.
g2p-en guessed the part of speech for a short list of such words instead, so "the refuse" and "to refuse" now sound the same.

## License and credits

The TTS engine in `src/tts_api/morshutalk/` is adapted from [MorshuTalk](https://github.com/n0spaces/MorshuTalk) by [n0spaces](https://github.com/n0spaces), released under the MIT License. Its license text is in `src/tts_api/morshutalk/LICENSE`.

The grapheme-to-phoneme model and code in `g2p.py` and `numbers.py` are adapted from [g2p-en](https://github.com/Kyubyong/g2p) by Kyubyong Park and Jongseok Kim, released under the Apache License 2.0. Its license text is in `g2p_data/model/LICENSE`.

The [CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict) is copyright Carnegie Mellon University and ships under its BSD-style license in `g2p_data/cmudict.LICENSE`. The copy in `g2p_data/cmudict.dict.gz` comes from commit `74790861` of that repository.

The source recording `morshu.wav` and the sprite frames in `morshutalk/sprites/` come from the CD-i game Link: The Faces of Evil.
They belong to their respective rights holders, and the MIT License of this repository does not cover them.

This project is licensed under the [MIT License](LICENSE).
You may use, change, and share it, as long as every copy keeps the copyright notice and the license text.
