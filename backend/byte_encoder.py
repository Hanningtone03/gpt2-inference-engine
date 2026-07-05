def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


BYTE_TO_UNICODE = bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}


def encode_bytes_to_unicode(text: str) -> str:
    raw_bytes = text.encode("utf-8")
    return "".join(BYTE_TO_UNICODE[b] for b in raw_bytes)


def decode_unicode_to_bytes(text: str) -> bytes:
    return bytes(UNICODE_TO_BYTE[ch] for ch in text)
