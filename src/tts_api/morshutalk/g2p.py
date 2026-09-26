# Progress-reporting wrapper around G2p.
# Core logic borrowed from G2p.__call__() — https://github.com/kyubyong/g2p (Apache-2.0)
from collections.abc import Callable

from g2p_en.g2p import (
    G2p,
    normalize_numbers,
    pos_tag,
    re,
    unicode,
    unicodedata,
    word_tokenize,
)


class G2pProgress(G2p):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def run_with_progress(
        self,
        text: str,
        callback: Callable[[int, int], None] | None = None,
    ) -> list[str]:
        self.cancelled = False

        text = unicode(text)
        text = normalize_numbers(text)
        text = "".join(
            char
            for char in unicodedata.normalize("NFD", text)
            if unicodedata.category(char) != "Mn"
        )
        text = text.lower()
        text = re.sub(r"[^ a-z'.,?!-]", "", text)
        text = text.replace("i.e.", "that is")
        text = text.replace("e.g.", "for example")

        words = word_tokenize(text)
        tokens = pos_tag(words)

        step = 0
        total = len(tokens)
        prons: list[str] = []

        for word, pos in tokens:
            if self.cancelled:
                return []
            if callback:
                callback(step, total)
                step += 1

            if re.search("[a-z]", word) is None:
                pron = [word]
            elif word in self.homograph2features:
                pron1, pron2, pos1 = self.homograph2features[word]
                pron = pron1 if pos.startswith(pos1) else pron2
            elif word in self.cmu:
                pron = self.cmu[word][0]
            else:
                pron = self.predict(word)

            prons.extend(pron)
            prons.extend([" "])

        if callback:
            callback(total, total)

        return prons[:-1]
