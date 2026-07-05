import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from bpe_tokenizer import BPETokenizer

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def make_tokenizer():
    return BPETokenizer(
        os.path.join(MODELS_DIR, "encoder.json"),
        os.path.join(MODELS_DIR, "vocab.bpe"),
    )


def test_encode_decode_round_trip_simple_sentence():
    tok = make_tokenizer()
    text = "Hello, world! This is a test."
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text


def test_known_token_id_for_common_word():
    tok = make_tokenizer()
    ids = tok.encode(" the")
    assert ids == [262]


def test_encode_produces_multiple_tokens_for_long_text():
    tok = make_tokenizer()
    text = "The quick brown fox jumps over the lazy dog."
    ids = tok.encode(text)
    assert len(ids) > 5


def test_empty_string_encodes_to_empty_list():
    tok = make_tokenizer()
    assert tok.encode("") == []


def test_round_trip_with_punctuation_and_numbers():
    tok = make_tokenizer()
    text = "GPT-2 has 124,000,000 parameters (approximately)."
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text


def test_round_trip_with_unicode_characters():
    tok = make_tokenizer()
    text = "café résumé naïve"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text
