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


def test_pieces_reconstruct_original_text():
    tok = make_tokenizer()
    text = "The quick brown fox"
    pieces = tok.encode_with_pieces(text)
    reconstructed = "".join(p["text"] for p in pieces)
    assert reconstructed == text


def test_pieces_match_plain_encode_ids():
    tok = make_tokenizer()
    text = "Hello, world!"
    pieces = tok.encode_with_pieces(text)
    plain_ids = tok.encode(text)
    assert [p["id"] for p in pieces] == plain_ids


def test_pieces_have_text_and_id_keys():
    tok = make_tokenizer()
    pieces = tok.encode_with_pieces("test")
    for p in pieces:
        assert "id" in p and "text" in p
