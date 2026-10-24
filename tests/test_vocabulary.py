"""Custom vocabulary: what is boosted, where a transcript has the words, and the guards.

Three words throughout: Haas (ハース), 有難う (ありがとう) and AIモデル (no reading)."""

import numpy as np
import pytest

from haas_asr import Vocabulary
from haas_asr.boosting import guarded, has_speech
from haas_asr.vocabulary import spellings

WORDS = {"Haas": ["ハース"], "有難う": ["ありがとう"], "AIモデル": []}


def test_words_are_cleaned():
    vocabulary = Vocabulary({" Haas ": ["ハース", " ハース", "Haas", ""], "有難う": "ありがとう"})
    assert vocabulary.words == {"Haas": ("ハース",), "有難う": ("ありがとう",)}
    assert not Vocabulary({})
    with pytest.raises(ValueError):
        Vocabulary([" "])


def test_words_parse_from_the_command_line_form():
    vocabulary = Vocabulary.parse(["Haas=ハース,はーす", "AIモデル", "Haas=ハウス"], alpha=3.0)
    assert vocabulary.words == {"Haas": ("ハース", "はーす", "ハウス"), "AIモデル": ()}
    assert vocabulary.alpha == 3.0


def test_a_word_is_boosted_as_the_model_can_write_it():
    assert spellings("有難う", ["ありがとう"]) == ["ありがとう", "アリガトウ", "有難う"]
    assert Vocabulary({"Haas": ["ハース"]}).spellings == ("ハース", "haas")


def test_words_are_found_whatever_kana_width_case_or_spacing_the_model_wrote():
    vocabulary = Vocabulary(WORDS)
    assert vocabulary.find("ＨＡＡＳ、アリガトウ") == [(0, 4, "Haas"), (5, 10, "有難う")]
    assert vocabulary.find("ai モデル") == [(0, 6, "AIモデル")]
    assert vocabulary.find("haasten") == []  # not inside a longer Latin word
    assert vocabulary.find("𠮷ハース") == [(1, 4, "Haas")]  # code points


def test_found_words_are_written_as_given():
    assert Vocabulary(WORDS).format("はーすさん、ありがとう。ai モデルです") == (
        "Haasさん、有難う。AIモデルです"
    )


def test_the_longer_word_wins():
    vocabulary = Vocabulary({"Haas": ["ハース"], "Haas-ja-v1.0": ["ハースjaバージョン1.0"]})
    assert vocabulary.format("ハースjaバージョン1.0とハース") == "Haas-ja-v1.0とHaas"


def test_the_guard_keeps_whole_listed_words_and_undoes_the_rest():
    words = Vocabulary({"Haas": ["ハース"]})
    assert guarded("はすです", "ハースです", words) == "ハースです"  # a fix: kept
    assert guarded("ハスの最新", "ハース最新", words) == "ハース最新"  # its spelling drops の
    assert guarded("そうですね", "そうハー", words) == "そうですね"  # unsupported: undone
    assert guarded("ハスのデモです", "ハースです", words) == "ハースのデモです"  # skipped speech


def test_silence_has_no_speech():
    assert not has_speech(np.full(16000, 0.005, np.float32))
    loud = np.zeros(16000, np.float32)
    loud[8000:8480] = 0.1
    assert has_speech(loud)
