"""Spells out numbers, money, and ordinals in English words before the phoneme conversion.

Adapted from g2p-en (https://github.com/Kyubyong/g2p, Apache-2.0).
g2p-en took it from keithito/tacotron (https://github.com/keithito/tacotron, MIT).
"""

import re
from typing import Any

import inflect

__all__ = ["normalize_numbers"]

_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")


def _words(number: int | str, **options: Any) -> str:
    """Returns inflect's English words for a number, always as one string."""
    # inflect accepts digits as a string, which its type hints describe better than an int.
    words = _inflect.number_to_words(str(number), **options)
    return " ".join(words) if isinstance(words, list) else words


def _remove_commas(m: re.Match[str]) -> str:
    return m.group(1).replace(",", "")


def _expand_decimal_point(m: re.Match[str]) -> str:
    return m.group(1).replace(".", " point ")


def _expand_dollars(m: re.Match[str]) -> str:
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        # The format is unexpected, so the number stays as it is.
        return match + " dollars"
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    dollar_unit = "dollar" if dollars == 1 else "dollars"
    cent_unit = "cent" if cents == 1 else "cents"
    if dollars and cents:
        return f"{dollars} {dollar_unit}, {cents} {cent_unit}"
    if dollars:
        return f"{dollars} {dollar_unit}"
    if cents:
        return f"{cents} {cent_unit}"
    return "zero dollars"


def _expand_ordinal(m: re.Match[str]) -> str:
    return _words(m.group(0))


def _expand_number(m: re.Match[str]) -> str:
    num = int(m.group(0))
    if 1000 < num < 3000:
        # Years such as 1999 are read as "nineteen ninety-nine".
        if num == 2000:
            return "two thousand"
        if 2000 < num < 2010:
            return "two thousand " + _words(num % 100)
        if num % 100 == 0:
            return _words(num // 100) + " hundred"
        return _words(num, andword="", zero="oh", group=2).replace(", ", " ")
    return _words(num, andword="")


def normalize_numbers(text: str) -> str:
    """Replaces every number in the text with its English words.

    Args:
        text: The text to rewrite.

    Returns:
        The text with numbers, money amounts, decimals, and ordinals spelled out.
    """
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    return re.sub(_number_re, _expand_number, text)
