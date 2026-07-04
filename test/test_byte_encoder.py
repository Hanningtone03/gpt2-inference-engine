import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from byte_encoder import bytes_to_unicode, encode_bytes_to_unicode, decode_unicode_to_bytes


def test_mapping_covers_all_256_byte_values():
    mapping = bytes_to_unicode()
    assert len(mapping) == 256
    assert set(mapping.keys()) == set(range(256))


def test_mapping_is_a_bijection():
    mapping = bytes_to_unicode()
    assert len(set(mapping.values())) == 256


def test_ascii_text_round_trips():
    text = "Hello, world!"
    encoded = encode_bytes_to_unicode(text)
    decoded = decode_unicode_to_bytes(encoded)
    assert decoded.decode("utf-8") == text


def test_unicode_text_round_trips():
    text = "héllo wörld 日本語"
    encoded = encode_bytes_to_unicode(text)
    decoded = decode_unicode_to_bytes(encoded)
    assert decoded.decode("utf-8") == text


def test_control_characters_round_trip():
    text = "line1\nline2\ttabbed"
    encoded = encode_bytes_to_unicode(text)
    decoded = decode_unicode_to_bytes(encoded)
    assert decoded.decode("utf-8") == text


def test_empty_string_round_trips():
    text = ""
    encoded = encode_bytes_to_unicode(text)
    decoded = decode_unicode_to_bytes(encoded)
    assert decoded.decode("utf-8") == text
