import random
import warnings
from collections.abc import Callable
from typing import Literal

import numpy as np
from pydub import AudioSegment

from .g2p import G2pProgress

# fmt: off
morshu_rec = np.rec.fromrecords([
    ('', 160, 0), ('L', 250, 2), ('AE', 348, 2), ('M', 420, 2), ('P', 510, 1),
    ('OY', 700, 2), ('L', 835, 1), ('', 1090, 0),
    ('R', 1180, 2), ('OW', 1300, 2), ('', 1390, 0), ('P', 1490, 2), ('', 1850, 0),
    ('B', 1895, 2), ('AA', 2090, 2), ('M', 2235, 2), ('Z', 2390, 2),
    ('', 2780, 0), ('Y', 2840, 2), ('UW', 2960, 2),
    ('W', 3030, 2), ('AA', 3110, 2), ('N', 3150, 1), ('IH', 3240, 2), ('T', 3370, 2), ('', 3810, 0),
    ('IH', 3960, 2), ('T', 4070, 2), ('Y', 4260, 2), ('UH', 4400, 2), ('R', 4510, 2), ('Z', 4600, 2),
    ('M', 4675, 2), ('AY', 4810, 2), ('', 4885, 0),
    ('F', 4930, 2), ('R', 4980, 2), ('EH', 5100, 2), ('N', 5240, 2), ('D', 5300, 2), ('', 5520, 0),
    ('AE', 5630, 2), ('Z', 5740, 2), ('L', 5870, 2), ('AO', 6000, 2), ('NG', 6140, 2),
    ('AE', 6170, 1), ('Z', 6265, 2), ('Y', 6300, 2), ('UW', 6380, 2),
    ('HH', 6450, 2), ('AE', 6510, 1), ('V', 6580, 2),
    ('IH', 6640, 2), ('N', 6670, 2), ('AH', 6747, 2), ('F', 6855, 2),
    ('R', 6960, 2), ('UW', 7060, 2), ('B', 7170, 1), ('IY', 7340, 2), ('Z', 7520, 2), ('', 8236, 0),
    ('S', 8407, 2), ('AA', 8495, 2), ('R', 8570, 2), ('IY', 8630, 1),
    ('L', 8740, 2), ('IH', 8811, 2), ('NG', 8942, 2), ('K', 9014, 2), ('', 9251, 0),
    ('AY', 9384, 2), ('', 9467, 0), ('K', 9512, 2), ('AE', 9640, 2), ('N', 9716, 2), ('', 9844, 0),
    ('G', 9894, 2), ('IH', 9985, 2), ('V', 10060, 2), ('', 10149, 0),
    ('K', 10256, 2), ('R', 10297, 2), ('EH', 10383, 2), ('IH', 10482, 1), ('', 10564, 0), ('T', 10617, 2),
    ('', 10962, 0), ('K', 11019, 2), ('AH', 11100, 2), ('M', 11229, 2), ('B', 11246, 2), ('AE', 11369, 2),
    ('', 11511, 0), ('W', 11590, 2), ('EH', 11622, 1), ('N', 11705, 2),
    ('Y', 11755, 2), ('UH', 11808, 2), ('R', 11864, 2), ('AH', 11959, 2),
    ('L', 12095, 2), ('IH', 12202, 2), ('L', 12386, 2),
    ('', 12596, 0), ('M', 12748, 2), ('M', 12888, 2), ('M', 13037, 2), ('M', 13196, 2), ('', 13426, 0),
    ('R', 13494, 2), ('IH', 13589, 2), ('', 13632, 0), ('CH', 13773, 2), ('ER', 13991, 2), ('', 13992, 0),
], names=('phoneme', 'timing', 'priority'))
# fmt: on

similar_phonemes: dict[str, list[str]] = {
    "AW": ["AE", "UW"],
    "DH": ["D"],
    "EY": ["EH", "IY"],
    "JH": ["CH"],
    "SH": ["CH"],
    "TH": ["D"],
    "ZH": ["CH"],
}

_g2p: G2pProgress | None = None
_morshu_wav: AudioSegment | None = None
_wav_path: str = "/data/morshu.wav"


def init(wav_path: str) -> None:
    """Load the G2p model and source WAV. Call once at application startup."""
    global _g2p, _morshu_wav, _wav_path
    import nltk

    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    _wav_path = wav_path
    _g2p = G2pProgress()
    _morshu_wav = AudioSegment.from_wav(wav_path)


def _ensure_loaded() -> None:
    if _g2p is None or _morshu_wav is None:
        init(_wav_path)


class Morshu:
    def __init__(self) -> None:
        self.input_str = ""
        self.input_phonemes: list[str] = []
        self.stop_chars = ".,?!:;()\n"
        self.space_length = 20
        self.stop_length = 100
        self.use_phoneme_priority = True
        self.out_audio = AudioSegment.empty()
        self.audio_segment_timings = np.rec.fromarrays(
            (0, 0), names=("output", "morshu")
        )
        self.canceled = False

    def cancel(self) -> None:
        if _g2p is not None:
            _g2p.cancel()
        self.canceled = True

    def load_text(
        self,
        text: str | None = None,
        progress_callback: Callable[[int, int, int], None] | None = None,
    ) -> AudioSegment | Literal[False]:
        _ensure_loaded()
        self.canceled = False

        if progress_callback is None:
            progress_callback = lambda major, minor, total: None  # noqa: E731

        if text is None:
            text = self.input_str
        self.input_str = text
        text = text.replace("\n", ",,,")

        assert _g2p is not None
        phonemes = _g2p.run_with_progress(
            text, lambda step, total: progress_callback(0, step, total)
        )
        if _g2p.cancelled:
            return False

        progress_step = 0
        progress_total = len(phonemes)

        assert _morshu_wav is not None
        output = AudioSegment.empty().set_frame_rate(_morshu_wav.frame_rate)
        audio_out_millis: list[int] = []
        audio_morshu_millis: list[int] = []

        phoneme_segment: list[str] = []
        while phonemes:
            if self.canceled:
                return False
            progress_callback(1, progress_step, progress_total)
            progress_step += 1

            p = phonemes.pop(0)
            if p in _g2p.phonemes:
                phoneme_segment.append(p)
            if p not in _g2p.phonemes or not phonemes:
                output = self.append_best_morshu_phoneme_segment(
                    output, phoneme_segment, audio_out_millis, audio_morshu_millis
                )
                phoneme_segment = []
            if p == " ":
                output = self.append_audio_segment(
                    output,
                    AudioSegment.silent(self.space_length),
                    -1,
                    audio_out_millis,
                    audio_morshu_millis,
                )
            elif p in self.stop_chars:
                output = self.append_audio_segment(
                    output,
                    AudioSegment.silent(self.stop_length),
                    -1,
                    audio_out_millis,
                    audio_morshu_millis,
                )

        if len(output) == 0:
            warnings.warn("returned audio segment is empty", UserWarning, stacklevel=2)
            self.audio_segment_timings = np.rec.fromarrays(
                (0, 0), names=("output", "morshu")
            )
        else:
            self.audio_segment_timings = np.rec.fromrecords(
                tuple(zip(audio_out_millis, audio_morshu_millis)),
                names=("output", "morshu"),
            )

        progress_callback(1, progress_total, progress_total)
        self.out_audio = output
        return output

    @staticmethod
    def substitute_similar_phonemes(phonemes: list[str]) -> list[str]:
        i = 0
        while i < len(phonemes):
            p = phonemes[i]
            if p.endswith(("0", "1", "2")):
                phonemes[i] = p[:-1]
                p = phonemes[i]
            if p in similar_phonemes:
                phonemes = phonemes[:i] + similar_phonemes[p] + phonemes[i + 1 :]
            i += 1
        return phonemes

    @staticmethod
    def append_audio_segment(
        audio_out: AudioSegment,
        audio_segment: AudioSegment,
        morshu_millis_start: int,
        audio_out_millis: list[int],
        audio_morshu_millis: list[int],
    ) -> AudioSegment:
        audio_out_millis.append(len(audio_out))
        audio_morshu_millis.append(morshu_millis_start)
        return audio_out + audio_segment

    @staticmethod
    def get_phoneme_sequence_occurrences(phonemes: list[str]) -> list[tuple[int, int]]:
        occurrences = []
        for i in range(len(morshu_rec) - len(phonemes)):
            if (morshu_rec["phoneme"][i : i + len(phonemes)] == phonemes).all():
                start = morshu_rec["timing"][i - 1]
                end = morshu_rec["timing"][i + len(phonemes) - 1]
                occurrences.append((int(start), int(end)))
        return occurrences

    def get_best_morshu_single_phoneme(
        self, phoneme: str, preceding: str = "", succeeding: str = ""
    ) -> tuple[AudioSegment, int]:
        assert _morshu_wav is not None
        best_indices: list[int] = []
        phoneme_indices = np.where(morshu_rec["phoneme"] == phoneme)[0]
        if len(phoneme_indices) == 0:
            return AudioSegment.empty(), 0

        highest_priority = 0
        for i in phoneme_indices:
            morshu_preceding = morshu_rec["phoneme"][i - 1]
            priority = (
                int(morshu_rec["priority"][i]) if self.use_phoneme_priority else 0
            )

            if morshu_preceding == preceding:
                priority += 10
            elif any(c in morshu_preceding for c in "AEIOU") and any(
                c in preceding for c in "AEIOU"
            ):
                priority += 5

            morshu_succeeding = morshu_rec["phoneme"][i + 1]
            if morshu_succeeding == succeeding:
                priority += 10
            elif any(c in morshu_succeeding for c in "AEIOU") and any(
                c in succeeding for c in "AEIOU"
            ):
                priority += 1

            if priority < highest_priority:
                continue
            if priority > highest_priority:
                highest_priority = priority
                best_indices = []
            best_indices.append(i)

        index = random.choice(best_indices)
        segment = _morshu_wav[
            int(morshu_rec["timing"][index - 1]) : int(morshu_rec["timing"][index])
        ]
        return segment, int(morshu_rec["timing"][index - 1])

    def append_best_morshu_phoneme_segment(
        self,
        output: AudioSegment,
        phonemes: list[str],
        audio_out_millis: list[int] | None = None,
        audio_morshu_millis: list[int] | None = None,
    ) -> AudioSegment:
        assert _morshu_wav is not None
        phonemes = Morshu.substitute_similar_phonemes(phonemes)
        if audio_out_millis is None:
            audio_out_millis = []
        if audio_morshu_millis is None:
            audio_morshu_millis = []

        if len(phonemes) == 1:
            segment, start = self.get_best_morshu_single_phoneme(phonemes[0])
            return Morshu.append_audio_segment(
                output, segment, start, audio_out_millis, audio_morshu_millis
            )

        preceding = ""
        while phonemes:
            sequence_length = 1
            segment = AudioSegment.empty()
            start = 0

            while sequence_length <= len(phonemes):
                occurrences = Morshu.get_phoneme_sequence_occurrences(
                    phonemes[:sequence_length]
                )
                if not occurrences:
                    break
                start, end = random.choice(occurrences)
                segment = _morshu_wav[start:end]
                sequence_length += 1
            sequence_length -= 1

            if sequence_length == 1:
                succeeding = (
                    phonemes[sequence_length] if sequence_length < len(phonemes) else ""
                )
                segment, start = self.get_best_morshu_single_phoneme(
                    phonemes[0], preceding, succeeding
                )

            output = Morshu.append_audio_segment(
                output, segment, start, audio_out_millis, audio_morshu_millis
            )
            preceding = phonemes[sequence_length - 1]
            del phonemes[:sequence_length]

        return output
