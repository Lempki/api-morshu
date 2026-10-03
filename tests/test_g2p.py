import gzip
from pathlib import Path

import pytest

from tts_api.morshutalk.g2p import G2p, load_dictionary


@pytest.fixture(scope="module")
def g2p() -> G2p:
    return G2p()


def _words(phonemes: list[str]) -> list[list[str]]:
    """Splits g2p output at the " " tokens and drops stress digits."""
    words: list[list[str]] = [[]]
    for p in phonemes:
        if p == " ":
            words.append([])
        else:
            words[-1].append(p.rstrip("012"))
    return words


def test_dictionary_reads_alternatives_and_skips_comments(tmp_path: Path) -> None:
    path = tmp_path / "dict.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(
            "read R IY1 D\nread(2) R EH1 D\nabc EY1 B IY1 S IY1 # abbrev\n\nbroken\n"
        )
    assert load_dictionary(path) == {
        "read": [["R", "IY1", "D"], ["R", "EH1", "D"]],
        "abc": [["EY1", "B", "IY1", "S", "IY1"]],
    }


def test_known_words_come_from_the_dictionary(g2p: G2p) -> None:
    assert _words(g2p("Lamp oil, rope, bombs")) == [
        ["L", "AE", "M", "P"],
        ["OY", "L"],
        [","],
        ["R", "OW", "P"],
        [","],
        ["B", "AA", "M", "Z"],
    ]


def test_contractions_and_hyphenated_words_stay_whole(g2p: G2p) -> None:
    words = _words(g2p("It's well-known."))
    assert words[0] == ["IH", "T", "S"]
    assert words[-1] == ["."]
    assert len(words) == 3


def test_unknown_words_are_predicted(g2p: G2p) -> None:
    # "morshu" is not in the dictionary, so the GRU model spells it out.
    assert "morshu" not in g2p._dictionary
    predicted = g2p.predict("morshu")
    assert predicted[0] == "M"
    assert g2p.predict("morshu") == predicted


def test_cancel_stops_the_next_conversion_step(g2p: G2p) -> None:
    calls: list[tuple[int, int]] = []

    def cancel_after_first_word(step: int, total: int) -> None:
        calls.append((step, total))
        if step == 1:
            g2p.cancel()

    assert g2p.run_with_progress("one two three", cancel_after_first_word) == []
    assert calls == [(0, 3), (1, 3)]
    assert g2p.run_with_progress("one") != []
