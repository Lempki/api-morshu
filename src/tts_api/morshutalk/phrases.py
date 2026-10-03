"""Finds the parts of a text that morshu.wav says word for word, so they can play as recorded.

The phoneme engine stitches speech from short clips, so even an exact quote comes out choppy.
A run of words that the recording says in the same order plays the recorded audio instead.
A run must have at least MIN_WORDS words and MIN_CHARS characters.
Shorter matches and every other word go to the phoneme engine, so it still does the creative work.
"""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "MIN_CHARS",
    "MIN_WORDS",
    "TRANSCRIPT",
    "Piece",
    "RecordedWord",
    "load_transcript",
    "split_text",
    "word_key",
]

MIN_WORDS = 2
MIN_CHARS = 8
TRANSCRIPT = Path(__file__).parent / "morshu_words.tsv"

# A word is a run of letters or digits, with inner apostrophes as in "it's".
# Digits count as words so that a number never hides between two pieces.
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
_HUMMING = re.compile(r"m{2,}")


@dataclass(frozen=True)
class RecordedWord:
    """One word of the recording.

    Attributes:
        key: The word in the form that matching compares, from word_key.
        start_ms: Where the word starts in morshu.wav.
        end_ms: Where the word ends in morshu.wav.
    """

    key: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class Piece:
    """A stretch of the input text, spoken either by the recording or by the phoneme engine.

    Attributes:
        text: The original text of the stretch, from its first word to its last.
        separator_before: The text between the previous piece and this one.
            For the first piece it is the text before the first word.
        recording: The start and end in morshu.wav, or None for the phoneme engine.
    """

    text: str
    separator_before: str
    recording: tuple[int, int] | None


def word_key(word: str) -> str:
    """Returns the form of a word that matching compares.

    Case, accents, and apostrophes are ignored, so "Its" matches "it's".
    A hum of two or more m's, such as "mm" or "mmmm", matches the recording's "mmm".

    Args:
        word: The word as typed.

    Returns:
        The comparable form.
    """
    folded = "".join(
        char
        for char in unicodedata.normalize("NFD", word.casefold())
        if unicodedata.category(char) != "Mn"
    )
    key = folded.replace("'", "").replace("’", "")
    return "mmm" if _HUMMING.fullmatch(key) else key


def load_transcript(path: Path = TRANSCRIPT) -> list[RecordedWord]:
    """Reads the word-level transcript of morshu.wav.

    Args:
        path: A tab-separated file of word, start, and end, where # starts a comment line.

    Returns:
        The words in the order the recording says them.
    """
    words: list[RecordedWord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        word, start, end = line.split("\t")
        words.append(RecordedWord(word_key(word), int(start), int(end)))
    return words


def split_text(text: str, transcript: list[RecordedWord]) -> tuple[list[Piece], str]:
    """Splits a text into recorded runs and stretches for the phoneme engine.

    The search is greedy from left to right.
    At each word it takes the longest run that the recording says in the same order.
    It keeps the run when the run is long enough, and otherwise hands the word to the engine.
    Punctuation inside a run does not break it, because the recording has its own pauses.

    Args:
        text: The text to speak.
        transcript: The words of the recording, from load_transcript.

    Returns:
        The pieces in order, and the text after the last word.
    """
    tokens = list(_WORD.finditer(text))
    keys = [word_key(token.group()) for token in tokens]
    spans: list[tuple[int, int, tuple[int, int] | None]] = []
    i = 0
    while i < len(tokens):
        length, first = _longest_match(keys, i, transcript)
        matched = keys[i : i + length]
        if length >= MIN_WORDS and len(" ".join(matched)) >= MIN_CHARS:
            span = (transcript[first].start_ms, transcript[first + length - 1].end_ms)
            spans.append((i, i + length - 1, span))
            i += length
            continue
        if spans and spans[-1][2] is None:
            spans[-1] = (spans[-1][0], i, None)
        else:
            spans.append((i, i, None))
        i += 1

    pieces: list[Piece] = []
    previous_end = 0
    for first_token, last_token, recorded in spans:
        start = tokens[first_token].start()
        end = tokens[last_token].end()
        pieces.append(Piece(text[start:end], text[previous_end:start], recorded))
        previous_end = end
    return pieces, text[previous_end:]


def _longest_match(
    keys: list[str], start: int, transcript: list[RecordedWord]
) -> tuple[int, int]:
    """Finds the longest run of the transcript that equals the words from start on.

    Returns:
        The run's length in words and its first index in the transcript.
        The first of several equally long runs wins.
    """
    best_length, best_first = 0, 0
    for first in range(len(transcript)):
        length = 0
        while (
            start + length < len(keys)
            and first + length < len(transcript)
            and keys[start + length] == transcript[first + length].key
        ):
            length += 1
        if length > best_length:
            best_length, best_first = length, first
    return best_length, best_first
