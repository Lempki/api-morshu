"""Tests for phrase matching inside the engine, with the real recording and G2p model."""

import random

import pytest

from tts_api.morshutalk import Morshu, init
from tts_api.morshutalk.g2p import load_dictionary
from tts_api.morshutalk.morshu import morshu_rec
from tts_api.morshutalk.phrases import load_transcript

TRANSCRIPT = load_transcript()
LINE_TWO = (
    "Sorry, Link. I can't give credit. Come back when you're a little, mmm, richer!"
)


def test_transcript_words_sound_like_their_spans() -> None:
    # Each word's phonemes in the table must share most sounds with its dictionary pronunciation.
    # A shifted span would pair a word with its neighbor's sounds and fail this.
    dictionary = load_dictionary()
    timings = [0, *(int(t) for t in morshu_rec["timing"])]
    for word in TRANSCRIPT:
        first = timings.index(word.start_ms)
        last = timings.index(word.end_ms)
        heard = [p for p in morshu_rec["phoneme"][first:last] if p]
        # The dictionary has no hum, and it spells "you're" with its apostrophe.
        pronunciation = {"mmm": [["M"]], "youre": dictionary["you're"]}.get(
            word.key, dictionary.get(word.key)
        )
        assert pronunciation is not None, word.key
        expected = [p.rstrip("012") for p in pronunciation[0]]
        shared = sum(1 for p in heard if p in expected)
        assert shared >= len(heard) / 2, (word.key, heard, expected)


@pytest.fixture(scope="module")
def engine() -> None:
    init(None)


@pytest.mark.usefixtures("engine")
def test_a_whole_line_is_one_clip_of_the_recording() -> None:
    m = Morshu()
    m.load_text(LINE_TWO)
    clips = [int(start) for start in m.audio_segment_timings["morshu"] if start >= 0]
    assert clips == [TRANSCRIPT[18].start_ms]


@pytest.mark.usefixtures("engine")
def test_phrase_matching_can_be_turned_off() -> None:
    m = Morshu(phrase_matching=False)
    m.load_text(LINE_TWO)
    assert sum(1 for start in m.audio_segment_timings["morshu"] if start >= 0) > 10


@pytest.mark.usefixtures("engine")
def test_text_without_a_match_is_spoken_exactly_as_before() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    random.seed(1)
    with_matching = Morshu().load_text(text)
    random.seed(1)
    without = Morshu(phrase_matching=False).load_text(text)
    assert with_matching is not False and without is not False
    assert with_matching.samples.tolist() == without.samples.tolist()


@pytest.mark.parametrize(
    ("separator", "between_words", "expected"),
    [
        (" ", True, 20),
        (", ", True, 140),
        ("!", False, 120),
        ("", False, 0),
        ("... ", True, 40),
    ],
)
def test_pause_lengths_follow_the_engine(
    separator: str, between_words: bool, expected: int
) -> None:
    assert Morshu().pause_length(separator, between_words) == expected
