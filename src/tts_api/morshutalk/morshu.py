"""The Morshu text-to-speech engine, adapted from MorshuTalk by n0spaces under the MIT License.

The engine converts text to phonemes.
It then stitches clips of the same phonemes, cut from morshu.wav, into new speech.
"""

import random
import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np

from .audio import Clip
from .g2p import G2p
from .phrases import Piece, load_transcript, split_text

# Each record is one phoneme of morshu.wav with its ARPAbet symbol, end time in ms, and priority.
# A phoneme starts where the previous record ends. An empty symbol marks silence.
# Single-phoneme matching prefers records with a higher priority.
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

# Phonemes that morshu.wav lacks, mapped to the phonemes that stand in for them.
similar_phonemes: dict[str, list[str]] = {
    "AW": ["AE", "UW"],
    "DH": ["D"],
    "EY": ["EH", "IY"],
    "JH": ["CH"],
    "SH": ["CH"],
    "TH": ["D"],
    "ZH": ["CH"],
}

# The source recording ships inside the package, so the service needs no mounted volume.
PACKAGED_WAV = Path(__file__).parent / "morshu.wav"

_g2p: G2p | None = None
_morshu_wav: Clip | None = None
_wav_path: Path = PACKAGED_WAV
_transcript = load_transcript()

# A punctuation token as g2p sees it. An ellipsis is one token, like in the G2p tokenizer.
_PUNCTUATION = re.compile(r"\.\.+|[^\s\w]")


def source_wav_path(wav_path: str | None) -> Path:
    """Resolves the source recording to load.

    Args:
        wav_path: A path that overrides the packaged recording, or None to use the packaged one.

    Returns:
        The path of the WAV file to load.
    """
    return PACKAGED_WAV if wav_path is None else Path(wav_path)


def init(wav_path: str | None = None) -> None:
    """Loads the G2p model, its dictionary, and the source WAV. Call it once at startup.

    Args:
        wav_path: A path that overrides the packaged morshu.wav, or None to use the packaged one.
    """
    global _g2p, _morshu_wav, _wav_path
    _wav_path = source_wav_path(wav_path)
    _g2p = G2p()
    _morshu_wav = Clip.from_wav(_wav_path)


def _ensure_loaded() -> None:
    if _g2p is None or _morshu_wav is None:
        init(str(_wav_path))


class Morshu:
    """Turns text into Morshu speech and remembers where each clip came from.

    Attributes:
        input_str: The text of the last conversion.
        stop_chars: The characters that insert a longer pause.
        space_length: The pause between words, in milliseconds.
        stop_length: The pause at a stop character, in milliseconds.
        use_phoneme_priority: Whether single-phoneme matching starts from each record's priority.
        out_audio: The audio of the last conversion.
        audio_segment_timings: One record per appended clip.
            "output" is where the clip starts in the output, in milliseconds.
            "morshu" is where it starts in morshu.wav, or -1 for an inserted pause.
        canceled: Whether cancel was called during the current conversion.
        phrase_matching: Whether word runs that morshu.wav says verbatim play as recorded.
    """

    def __init__(self, phrase_matching: bool = True) -> None:
        self.input_str = ""
        self.input_phonemes: list[str] = []
        self.stop_chars = ".,?!:;()\n"
        self.space_length = 20
        self.stop_length = 100
        self.use_phoneme_priority = True
        self.out_audio: Clip | None = None
        self.audio_segment_timings = np.rec.fromarrays(
            (0, 0), names=("output", "morshu")
        )
        self.canceled = False
        self.phrase_matching = phrase_matching

    def cancel(self) -> None:
        """Stops the running conversion, including the phoneme conversion."""
        if _g2p is not None:
            _g2p.cancel()
        self.canceled = True

    def load_text(
        self,
        text: str | None = None,
        progress_callback: Callable[[int, int, int], None] | None = None,
    ) -> Clip | Literal[False]:
        """Converts text to Morshu speech and records where each clip came from.

        It loads the model and the source WAV first if init has not run yet.
        With phrase_matching on, word runs that morshu.wav says verbatim play as recorded.
        The phoneme engine speaks everything else.

        Args:
            text: The text to speak, or None to repeat the last text.
            progress_callback: Called with a stage, a step, and a total.
                Stage 0 is the phoneme conversion and stage 1 is the audio stitching.
                With phrase matching, the stages restart for each stretch the engine speaks.

        Returns:
            The stitched audio, or False when the conversion was cancelled.
        """
        _ensure_loaded()
        assert _morshu_wav is not None
        self.canceled = False

        if progress_callback is None:
            progress_callback = lambda major, minor, total: None  # noqa: E731

        if text is None:
            text = self.input_str
        self.input_str = text
        text = text.replace("\n", ",,,")

        pieces, trailing = (
            split_text(text, _transcript) if self.phrase_matching else ([], "")
        )
        if not any(piece.recording for piece in pieces):
            # Without a recorded run the whole text goes to the engine, exactly as before.
            pieces, trailing = [Piece(text, "", None)], ""

        output = Clip.empty(_morshu_wav.frame_rate)
        audio_out_millis: list[int] = []
        audio_morshu_millis: list[int] = []
        for index, piece in enumerate(pieces):
            if self.canceled:
                return False
            output = self._append_pause(
                output,
                piece.separator_before,
                index > 0,
                audio_out_millis,
                audio_morshu_millis,
            )
            if piece.recording is not None:
                start, end = piece.recording
                output = self.append_audio_segment(
                    output,
                    _morshu_wav[start:end],
                    start,
                    audio_out_millis,
                    audio_morshu_millis,
                )
                continue
            spoken = self._speak(
                piece.text,
                output,
                audio_out_millis,
                audio_morshu_millis,
                progress_callback,
            )
            if spoken is False:
                return False
            output = spoken
        output = self._append_pause(
            output, trailing, False, audio_out_millis, audio_morshu_millis
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

        self.out_audio = output
        return output

    def _speak(
        self,
        text: str,
        output: Clip,
        audio_out_millis: list[int],
        audio_morshu_millis: list[int],
        progress_callback: Callable[[int, int, int], None],
    ) -> Clip | Literal[False]:
        """Appends the phoneme engine's speech for a text.

        Args:
            text: The text to speak.
            output: The output so far.
            audio_out_millis: The output start times, which receives one entry per clip.
            audio_morshu_millis: The morshu.wav start times, which receives one entry per clip.
            progress_callback: Called with a stage, a step, and a total.

        Returns:
            The output with the speech appended, or False when the conversion was cancelled.
        """
        assert _g2p is not None
        assert _morshu_wav is not None
        phonemes = _g2p.run_with_progress(
            text, lambda step, total: progress_callback(0, step, total)
        )
        if _g2p.cancelled:
            return False

        progress_step = 0
        progress_total = len(phonemes)
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
                    Clip.silent(self.space_length, _morshu_wav.frame_rate),
                    -1,
                    audio_out_millis,
                    audio_morshu_millis,
                )
            elif p in self.stop_chars:
                output = self.append_audio_segment(
                    output,
                    Clip.silent(self.stop_length, _morshu_wav.frame_rate),
                    -1,
                    audio_out_millis,
                    audio_morshu_millis,
                )

        progress_callback(1, progress_total, progress_total)
        return output

    def pause_length(self, separator: str, between_words: bool) -> int:
        """Returns how long the engine would pause for the text between two words.

        The engine pauses for space_length at every token boundary.
        It pauses for stop_length more at each stop character.

        Args:
            separator: The text between two pieces, such as ", " or " ".
            between_words: Whether a word comes before the separator.
                Only then does the boundary after that word add a pause.

        Returns:
            The pause in milliseconds.
        """
        marks = _PUNCTUATION.findall(separator)
        boundaries = len(marks) + (1 if between_words else 0)
        stops = sum(1 for mark in marks if mark in self.stop_chars)
        return boundaries * self.space_length + stops * self.stop_length

    def _append_pause(
        self,
        output: Clip,
        separator: str,
        between_words: bool,
        audio_out_millis: list[int],
        audio_morshu_millis: list[int],
    ) -> Clip:
        """Appends the pause for a separator, or nothing when it calls for none."""
        assert _morshu_wav is not None
        length = self.pause_length(separator, between_words)
        if length == 0:
            return output
        return self.append_audio_segment(
            output,
            Clip.silent(length, _morshu_wav.frame_rate),
            -1,
            audio_out_millis,
            audio_morshu_millis,
        )

    @staticmethod
    def substitute_similar_phonemes(phonemes: list[str]) -> list[str]:
        """Removes stress digits and replaces phonemes that morshu.wav lacks.

        Args:
            phonemes: The phonemes to rewrite. Stress digits are removed from this list in place.

        Returns:
            The phonemes with every missing phoneme replaced by its stand-ins.
        """
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
        audio_out: Clip,
        audio_segment: Clip,
        morshu_millis_start: int,
        audio_out_millis: list[int],
        audio_morshu_millis: list[int],
    ) -> Clip:
        """Appends a clip to the output and records where it starts.

        Args:
            audio_out: The output so far.
            audio_segment: The clip to append.
            morshu_millis_start: Where the clip starts in morshu.wav, or -1 for a pause.
            audio_out_millis: The output start times, which receives the clip's start.
            audio_morshu_millis: The morshu.wav start times, which receives morshu_millis_start.

        Returns:
            The output with the clip appended.
        """
        audio_out_millis.append(len(audio_out))
        audio_morshu_millis.append(morshu_millis_start)
        return audio_out + audio_segment

    @staticmethod
    def get_phoneme_sequence_occurrences(phonemes: list[str]) -> list[tuple[int, int]]:
        """Finds every place where morshu.wav says the phonemes in a row.

        Args:
            phonemes: The phoneme sequence to find.

        Returns:
            The start and end of each occurrence in morshu.wav, in milliseconds.
        """
        occurrences = []
        for i in range(len(morshu_rec) - len(phonemes)):
            if (morshu_rec["phoneme"][i : i + len(phonemes)] == phonemes).all():
                start = morshu_rec["timing"][i - 1]
                end = morshu_rec["timing"][i + len(phonemes) - 1]
                occurrences.append((int(start), int(end)))
        return occurrences

    def get_best_morshu_single_phoneme(
        self, phoneme: str, preceding: str = "", succeeding: str = ""
    ) -> tuple[Clip, int]:
        """Picks the clip of one phoneme whose neighbors in morshu.wav fit best.

        A clip scores higher when its neighbors equal the given ones, or when both are vowels.
        A random clip among the best scores is chosen.

        Args:
            phoneme: The phoneme to find.
            preceding: The phoneme before it in the output, or "" for none.
            succeeding: The phoneme after it in the output, or "" for none.

        Returns:
            The clip and where it starts in morshu.wav, in milliseconds.
            The clip is empty when morshu.wav never says the phoneme.
        """
        assert _morshu_wav is not None
        best_indices: list[int] = []
        phoneme_indices = np.where(morshu_rec["phoneme"] == phoneme)[0]
        if len(phoneme_indices) == 0:
            return Clip.empty(_morshu_wav.frame_rate), 0

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
        output: Clip,
        phonemes: list[str],
        audio_out_millis: list[int] | None = None,
        audio_morshu_millis: list[int] | None = None,
    ) -> Clip:
        """Appends the audio of one word, built from the longest phoneme runs that morshu.wav holds.

        Args:
            output: The output so far.
            phonemes: The phonemes of the word. The list is emptied as they are used.
            audio_out_millis: The output start times, which receives one entry per clip.
            audio_morshu_millis: The morshu.wav start times, which receives one entry per clip.

        Returns:
            The output with the word appended.
        """
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
            segment = Clip.empty(_morshu_wav.frame_rate)
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
