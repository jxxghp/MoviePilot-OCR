import io
import unittest
from unittest.mock import patch

from PIL import Image

import main
from main import (
    ImageTooLargeError,
    InvalidImageError,
    decode_base64_image,
    load_grayscale_image,
    normalize_prediction,
    recognize_captcha,
)


a = {
    "base64_img": "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"
}

image_b64 = "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"
b = {"base64_img": image_b64}


class CaptchaRecognitionTest(unittest.TestCase):
    """覆盖验证码识别回归、输入校验和候选选择策略。"""

    def test_repository_samples(self):
        """仓库内已标注样本必须保持原有正确结果。"""

        samples = (
            (a["base64_img"], "77D2A8"),
            (b["base64_img"], "3BME4D"),
        )

        for encoded, expected in samples:
            with self.subTest(expected=expected):
                image_bytes = decode_base64_image(encoded)
                self.assertEqual(recognize_captcha(image_bytes), expected)

    def test_data_url_and_missing_padding_are_accepted(self):
        """兼容 data URL 和省略 Base64 尾部填充的调用方。"""

        encoded = a["base64_img"].rstrip("=")
        image_bytes = decode_base64_image(f"data:image/png;base64,{encoded}")
        self.assertTrue(image_bytes.startswith(b"\x89PNG"))

    def test_invalid_base64_is_rejected(self):
        """非法 Base64 必须作为客户端输入错误拒绝。"""

        with self.assertRaisesRegex(InvalidImageError, "not valid Base64"):
            decode_base64_image("not-base64")

    def test_prediction_is_normalized_and_requires_six_characters(self):
        """只接受六位 ASCII 字母数字，并统一转换为大写。"""

        self.assertEqual(normalize_prediction("a1B2c3"), "A1B2C3")
        self.assertEqual(normalize_prediction("a1-b2c3"), "A1B2C3")
        self.assertEqual(normalize_prediction("ABCDE"), "")
        self.assertEqual(normalize_prediction("中文123456"), "123456")

    def test_candidate_voting_overrides_a_single_primary_mistake(self):
        """多个候选一致时应纠正旧预处理路径的单次误判。"""

        candidates = [b"primary", b"threshold", b"line-removed"]
        with (
            patch("main.build_image_candidates", return_value=candidates),
            patch(
                "main.classify_candidate",
                side_effect=(("O9NNMR", 0.7), ("D9NNMR", 0.9), ("D9NNMR", 0.8)),
            ),
        ):
            self.assertEqual(recognize_captcha(b"image"), "D9NNMR")

    def test_candidate_tie_keeps_primary_result(self):
        """候选票数相同时保留旧路径结果，防止已有站点回归。"""

        with (
            patch(
                "main.build_image_candidates",
                return_value=[b"primary", b"fallback"],
            ),
            patch(
                "main.classify_candidate",
                side_effect=(("77D2A8", 0.9), ("77DZA8", 0.9)),
            ),
        ):
            self.assertEqual(recognize_captcha(b"image"), "77D2A8")

    def test_low_confidence_non_od_conflict_keeps_primary_result(self):
        """低置信度候选若不是已确认的 O/D 混淆，仍保留基线结果。"""

        with (
            patch(
                "main.build_image_candidates",
                return_value=[b"primary", b"fallback-1", b"fallback-2"],
            ),
            patch(
                "main.classify_candidate",
                side_effect=(
                    ("49DEDG", 0.8),
                    ("49DEDC", 0.95),
                    ("49DEDC", 0.9),
                ),
            ),
        ):
            self.assertEqual(recognize_captcha(b"image"), "49DEDG")

    def test_transparent_image_is_composited_on_white(self):
        """透明图片必须先铺白底，避免透明像素被误当成黑色字符。"""

        image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        gray = load_grayscale_image(buffer.getvalue())

        self.assertTrue((gray == 255).all())

    def test_invalid_image_content_is_rejected(self):
        """可解码 Base64 中的非图片内容必须返回图片格式错误。"""

        with self.assertRaisesRegex(InvalidImageError, "not a supported image"):
            load_grayscale_image(b"not-an-image")

    def test_oversized_image_dimensions_are_rejected(self):
        """高压缩率图片也必须执行像素尺寸限制，防止解压炸弹。"""

        image = Image.new("RGB", (4097, 1), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        with self.assertRaisesRegex(ImageTooLargeError, "4096 x 4096"):
            load_grayscale_image(buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
