"""
TTS tests - sentence splitting (pure/fast) and real Piper synthesis
(slower, uses the `tts` session fixture so the model loads only once).
"""

from __future__ import annotations

from app.tts import split_into_sentences


def test_split_simple_sentences():
    assert split_into_sentences("Hello there. How are you?") == ["Hello there.", "How are you?"]


def test_split_handles_exclamation():
    assert split_into_sentences("Watch out!") == ["Watch out!"]


def test_split_empty_string_returns_empty_list():
    assert split_into_sentences("") == []
    assert split_into_sentences("   ") == []


def test_split_no_terminal_punctuation_returns_single_fragment():
    assert split_into_sentences("this has no ending") == ["this has no ending"]


def test_synthesize_pcm_produces_nonempty_16bit_audio(tts):
    pcm = tts.synthesize_pcm("This is a test.")
    assert len(pcm) > 0
    assert len(pcm) % 2 == 0  # 16-bit samples -> even byte count


def test_synthesize_pcm_duration_roughly_matches_text_length(tts):
    short = tts.synthesize_pcm("Hi.")
    long = tts.synthesize_pcm("This is a considerably longer sentence than the short one.")
    assert len(long) > len(short)
