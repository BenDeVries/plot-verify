"""Pin down current color-helper behavior in app_auto_axis.py.

These tests run BEFORE Refactor A. After the move into plotverify_core.colors,
the corresponding tests in tests/test_colors.py must produce identical values.
"""
import pytest

from app_auto_axis import (
    is_valid_hex,
    hex_to_hsv_opencv,
    hex_to_bgr,
    hex_complement,
)


class TestIsValidHex:
    @pytest.mark.parametrize("value", ["#ff0000", "ff0000", "#FF00aa", "abcdef"])
    def test_valid(self, value):
        assert is_valid_hex(value) is True

    @pytest.mark.parametrize(
        "value",
        ["#fff", "ff00", "#gghhii", "", None, 123, "#1234567", "abcd"],
    )
    def test_invalid(self, value):
        assert is_valid_hex(value) is False


class TestHexConversions:
    def test_hex_to_hsv_red(self):
        h, s, v = hex_to_hsv_opencv("#ff0000")
        assert h == 0
        assert s == 255
        assert v == 255

    def test_hex_to_hsv_white(self):
        h, s, v = hex_to_hsv_opencv("#ffffff")
        assert (h, s, v) == (0, 0, 255)

    def test_hex_to_hsv_black(self):
        assert hex_to_hsv_opencv("#000000") == (0, 0, 0)

    def test_hex_to_bgr_red(self):
        # BGR ordering: red is (0, 0, 255).
        assert hex_to_bgr("#ff0000") == (0, 0, 255)

    def test_hex_to_bgr_blue(self):
        assert hex_to_bgr("#0000ff") == (255, 0, 0)

    def test_hex_to_bgr_invalid_returns_grey(self):
        assert hex_to_bgr("not-a-color") == (136, 136, 136)


class TestHexComplement:
    def test_complement_bright_red_is_cyan_like(self):
        # Bright red's hue-opposite has hue ~0.5; we just check it's not red.
        c = hex_complement("#ff0000")
        assert c.startswith("#")
        assert c.lower() != "#ff0000"

    def test_complement_returns_valid_hex(self):
        for h in ("#ff0000", "#00ff00", "#0000ff", "#888888", "#020202"):
            c = hex_complement(h)
            assert is_valid_hex(c), f"hex_complement({h}) returned invalid {c}"

    def test_complement_of_dark_color_is_light(self):
        # The dark branch is intentionally non-deterministic for contrast.
        # We assert only that the result decodes as a "light" RGB (value > 0.5).
        import colorsys

        c = hex_complement("#020202")
        h = c.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        _, _, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        assert v >= 0.85, f"dark-input complement should be light, got v={v}"
