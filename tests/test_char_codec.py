from virgo2.lm.char_codec import CharCodec


def test_codec_roundtrip() -> None:
    codec = CharCodec()
    codec.fit(["hello"])
    ids = codec.encode("hello")
    assert codec.decode(ids) == "hello"
