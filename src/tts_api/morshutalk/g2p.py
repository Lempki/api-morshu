"""Converts English text to ARPAbet phonemes, reporting progress and allowing cancellation.

Adapted from g2p-en by Kyubyong Park and Jongseok Kim (https://github.com/Kyubyong/g2p).
g2p-en is licensed under Apache-2.0, and g2p_data/model/LICENSE holds its license text.
This version needs no NLTK, and it changes g2p-en in three ways.
Pronunciations come from the CMU Pronouncing Dictionary in g2p_data/cmudict.dict.gz.
Words are split with a regular expression instead of NLTK's TweetTokenizer.
A word with several pronunciations takes the dictionary's first one.
g2p-en guessed the part of speech for a list of homographs instead, which needed NLTK's tagger.
The neural model with g2p-en's original weights still predicts words the dictionary lacks.
"""

import gzip
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .numbers import normalize_numbers

__all__ = ["G2p"]

_DATA = Path(__file__).parent / "g2p_data"
_DICTIONARY = _DATA / "cmudict.dict.gz"
_MODEL = _DATA / "model"

Array = npt.NDArray[np.float32]

_GRAPHEMES = ["<pad>", "<unk>", "</s>", *"abcdefghijklmnopqrstuvwxyz"]
# fmt: off
_PHONEMES = ["<pad>", "<unk>", "<s>", "</s>"] + [
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2",
    "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH0", "EH1", "EH2",
    "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH", "IH0", "IH1", "IH2", "IY0",
    "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2",
    "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W",
    "Y", "Z", "ZH",
]
# fmt: on

# A word keeps inner apostrophes and hyphens, as in "it's" and "well-known".
# An ellipsis stays one token, and any other character becomes a token of its own.
_TOKEN = re.compile(r"[a-z]+(?:['\-][a-z]+)*|\.\.+|[^\sa-z]")

# The model predicts at most this many phonemes for one word.
_MAX_PREDICTED = 20


def load_dictionary(path: Path = _DICTIONARY) -> dict[str, list[list[str]]]:
    """Reads the CMU Pronouncing Dictionary.

    Args:
        path: The gzip-compressed dictionary in the cmusphinx format.

    Returns:
        Each lowercase word mapped to its pronunciations, the primary one first.
    """
    pronunciations: dict[str, list[list[str]]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as lines:
        for line in lines:
            # Some entries end in a comment, such as "# abbrev".
            fields = line.split("#", 1)[0].split()
            if len(fields) < 2:
                continue
            # Alternative pronunciations are listed as "word(2)", "word(3)", and so on.
            word = fields[0].split("(", 1)[0]
            pronunciations.setdefault(word, []).append(fields[1:])
    return pronunciations


def _sigmoid(x: Array) -> Array:
    result: Array = (1 / (1 + np.exp(-x))).astype(np.float32)
    return result


class G2p:
    """Converts text to phonemes with the CMU dictionary and a GRU model for unknown words.

    Attributes:
        phonemes: Every phoneme symbol the model knows, with stress digits.
        cancelled: Whether cancel was called during the current conversion.
    """

    def __init__(self) -> None:
        self.phonemes = _PHONEMES
        self.cancelled = False
        self._g2idx = {g: i for i, g in enumerate(_GRAPHEMES)}
        self._dictionary = load_dictionary()
        self._weights = {
            path.stem: np.load(path).astype(np.float32) for path in _MODEL.glob("*.npy")
        }

    def cancel(self) -> None:
        """Stops the running conversion before its next word."""
        self.cancelled = True

    def __call__(self, text: str) -> list[str]:
        """Converts text to phonemes without progress reports."""
        return self.run_with_progress(text)

    def run_with_progress(
        self,
        text: str,
        callback: Callable[[int, int], None] | None = None,
    ) -> list[str]:
        """Converts text to ARPAbet phonemes, one word at a time.

        Args:
            text: The text to convert.
            callback: Called with the number of words done and the total word count.
                It runs before each word and once more at the end.

        Returns:
            The phonemes, with a " " token between words and punctuation kept as tokens.
            The list is empty when the conversion was cancelled.
        """
        self.cancelled = False
        tokens = _TOKEN.findall(_clean(text))
        total = len(tokens)
        prons: list[str] = []
        for step, word in enumerate(tokens):
            if self.cancelled:
                return []
            if callback:
                callback(step, total)
            if re.search("[a-z]", word) is None:
                prons.append(word)
            elif word in self._dictionary:
                prons.extend(self._dictionary[word][0])
            else:
                prons.extend(self.predict(word))
            prons.append(" ")
        if callback:
            callback(total, total)
        return prons[:-1]

    def predict(self, word: str) -> list[str]:
        """Predicts the phonemes of a word that the dictionary lacks.

        Args:
            word: The lowercase word.

        Returns:
            The predicted phonemes, with stress digits.
        """
        w = self._weights
        chars = [*word, "</s>"]
        ids = [self._g2idx.get(char, self._g2idx["<unk>"]) for char in chars]
        h = np.zeros((1, w["enc_w_hh"].shape[1]), np.float32)
        for index in ids:
            x = w["enc_emb"][[index]]
            h = self._gru_cell(x, h, "enc")

        predicted: list[str] = []
        x = w["dec_emb"][[2]]  # Index 2 is the <s> start symbol.
        for _ in range(_MAX_PREDICTED):
            h = self._gru_cell(x, h, "dec")
            logits = h @ w["fc_w"].T + w["fc_b"]
            index = int(logits.argmax())
            if index == 3:  # Index 3 is the </s> end symbol.
                break
            predicted.append(_PHONEMES[index])
            x = w["dec_emb"][[index]]
        return predicted

    def _gru_cell(self, x: Array, h: Array, prefix: str) -> Array:
        """Runs one step of the encoder's or the decoder's GRU, as g2p-en's grucell does."""
        w = self._weights
        rzn_ih = x @ w[f"{prefix}_w_ih"].T + w[f"{prefix}_b_ih"]
        rzn_hh = h @ w[f"{prefix}_w_hh"].T + w[f"{prefix}_b_hh"]
        split = rzn_ih.shape[-1] * 2 // 3
        rz = _sigmoid(rzn_ih[:, :split] + rzn_hh[:, :split])
        r, z = np.split(rz, 2, -1)
        n = np.tanh(rzn_ih[:, split:] + r * rzn_hh[:, split:])
        result: Array = (1 - z) * n + z * h
        return result


def _clean(text: str) -> str:
    """Spells out numbers, strips accents, lowercases, and keeps only the characters g2p reads."""
    text = normalize_numbers(text)
    text = "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )
    text = text.lower()
    text = re.sub(r"[^ a-z'.,?!-]", "", text)
    text = text.replace("i.e.", "that is")
    return text.replace("e.g.", "for example")
