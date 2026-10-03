import pytest

from tts_api.morshutalk.numbers import normalize_numbers


def test_numbers_are_spelled_out() -> None:
    assert normalize_numbers("3 bombs, $250, 1999, 21st") == (
        "three bombs, two hundred fifty dollars, nineteen ninety-nine, twenty-first"
    )


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("2000", "two thousand"),
        ("2005", "two thousand five"),
        ("1900", "nineteen hundred"),
        ("$1.01", "one dollar, one cent"),
        ("£12", "twelve pounds"),
        ("3.14", "three point fourteen"),
        ("1,000,000", "one million"),
    ],
)
def test_number_forms(text: str, spoken: str) -> None:
    assert normalize_numbers(text) == spoken
