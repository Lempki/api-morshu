import pytest

from tts_api.morshutalk.morshu import morshu_rec
from tts_api.morshutalk.phrases import (
    MIN_CHARS,
    MIN_WORDS,
    load_transcript,
    split_text,
    word_key,
)

TRANSCRIPT = load_transcript()
LINE_ONE = (
    "Lamp oil, rope, bombs, you want it? It's yours, my friend, "
    "as long as you have enough rubies."
)
LINE_TWO = (
    "Sorry, Link. I can't give credit. Come back when you're a little, mmm, richer!"
)


def _shape(text: str) -> list[tuple[str, bool]]:
    """Each piece's text and whether it plays as recorded."""
    pieces, _ = split_text(text, TRANSCRIPT)
    return [(piece.text, piece.recording is not None) for piece in pieces]


# The transcript of morshu.wav.


def test_transcript_is_in_order_and_on_phoneme_boundaries() -> None:
    boundaries = {0, *(int(t) for t in morshu_rec["timing"])}
    previous_end = 0
    for word in TRANSCRIPT:
        assert word.start_ms in boundaries and word.end_ms in boundaries, word
        assert previous_end <= word.start_ms < word.end_ms, word
        previous_end = word.end_ms


def test_transcript_covers_both_lines() -> None:
    keys = " ".join(word.key for word in TRANSCRIPT)
    assert keys == " ".join(
        word_key(w)
        for w in (LINE_ONE + " " + LINE_TWO)
        .replace(",", " ")
        .replace(".", " ")
        .replace("?", " ")
        .replace("!", " ")
        .split()
    )


# Matching.


@pytest.mark.parametrize(
    ("typed", "key"),
    [("It's", "its"), ("CAN’T", "cant"), ("mm", "mmm"), ("Mmmmm", "mmm"), ("m", "m")],
)
def test_word_key(typed: str, key: str) -> None:
    assert word_key(typed) == key


def test_new_text_and_a_recorded_run_mix() -> None:
    # The first "my" is a single word, so only the second "my friend" is long enough to match.
    assert _shape("Welcome to my shop, my friend!") == [
        ("Welcome to my shop", False),
        ("my friend", True),
    ]


def test_whole_lines_play_as_one_recording() -> None:
    assert _shape(LINE_ONE) == [(LINE_ONE[:-1], True)]
    assert _shape(LINE_TWO) == [(LINE_TWO[:-1], True)]


def test_runs_match_in_any_order() -> None:
    assert _shape("Rope, bombs, lamp oil.") == [
        ("Rope, bombs", True),
        ("lamp oil", True),
    ]


@pytest.mark.parametrize(
    ("text", "recorded"),
    [
        ("you want", True),  # Two words and eight characters, which is exactly enough.
        ("as long", False),  # Two words but seven characters.
        ("rubies", False),  # One word, however long.
        ("you're a", False),  # Seven characters once the apostrophe is ignored.
    ],
)
def test_minimum_run_length(text: str, recorded: bool) -> None:
    assert MIN_WORDS == 2 and MIN_CHARS == 8
    assert any(is_recorded for _, is_recorded in _shape(text)) is recorded


def test_numbers_stay_with_the_engine() -> None:
    assert _shape("lamp oil 3 rope bombs") == [
        ("lamp oil", True),
        ("3", False),
        ("rope bombs", True),
    ]


def test_separators_and_trailing_text_are_kept() -> None:
    pieces, trailing = split_text("Well... sorry Link!!", TRANSCRIPT)
    assert [p.separator_before for p in pieces] == ["", "... "]
    assert trailing == "!!"
