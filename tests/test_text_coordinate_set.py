from virgo2.ddif.coordinate_set import TextCoordinateSet
from virgo2.lm.char_codec import CharCodec


def test_coordinate_shapes() -> None:
    codec = CharCodec()
    codec.fit(["hello"])
    cs = TextCoordinateSet(codec)
    coords, targets = cs.from_text("hello")
    assert coords.shape == (5, 1)
    assert targets.shape == (5,)
