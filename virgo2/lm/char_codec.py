from __future__ import annotations


class CharCodec:
    def __init__(self) -> None:
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: list[str] = []

    def fit(self, texts: list[str]) -> None:
        vocab = sorted(set("".join(texts)))
        self.id_to_char = vocab
        self.char_to_id = {ch: i for i, ch in enumerate(vocab)}

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[ch] for ch in text if ch in self.char_to_id]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_char)
