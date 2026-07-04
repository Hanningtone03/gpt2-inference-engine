import json
import regex as re
from byte_encoder import encode_bytes_to_unicode, decode_unicode_to_bytes


GPT2_SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class BPETokenizer:
    def __init__(self, encoder_path, vocab_bpe_path):
        with open(encoder_path, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(vocab_bpe_path, "r", encoding="utf-8") as f:
            bpe_data = f.read()
        merges = [tuple(line.split()) for line in bpe_data.split("\n")[1:-1]]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        self.cache = {}

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            min_pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if min_pair not in self.bpe_ranks:
                break

            first, second = min_pair
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text):
        token_ids = []
        for chunk in GPT2_SPLIT_PATTERN.findall(text):
            chunk_bytes_encoded = encode_bytes_to_unicode(chunk)
            for bpe_token in self.bpe(chunk_bytes_encoded).split(" "):
                token_ids.append(self.encoder[bpe_token])
        return token_ids

    def decode(self, token_ids):
        text = "".join(self.decoder[t] for t in token_ids)
        return decode_unicode_to_bytes(text).decode("utf-8", errors="replace")
