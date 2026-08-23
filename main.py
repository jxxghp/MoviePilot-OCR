import base64
import binascii
import io
import re
from threading import Lock

import cv2
import ddddocr
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, status
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
DARK_PIXEL_THRESHOLD = 64
MIN_COMPONENT_AREA = 3
BORDER_WIDTH = 2
CAPTCHA_PATTERN = re.compile(r"^[A-Z0-9]{6}$")
PRIMARY_CONFIDENCE_THRESHOLD = 0.84

app = FastAPI(title="MoviePilot OCR", version="2.0.0")

# The beta model is materially more accurate on MoviePilot's captcha samples.
ocr = ddddocr.DdddOcr(beta=True, show_ad=False)
ocr_lock = Lock()


class OCRRequest(BaseModel):
    """验证码识别请求，包含纯 Base64 或 data URL 图片。"""

    base64_img: str


class OCRResponse(BaseModel):
    """验证码识别响应。"""

    result: str


class ImageTooLargeError(ValueError):
    """图片超过字节数或像素尺寸限制。"""

    pass


class InvalidImageError(ValueError):
    """请求内容不是可解码的有效图片。"""

    pass


@app.get("/")
def root():
    """返回服务存活状态。"""

    return {"message": "MoviePilot OCR API"}


@app.post("/captcha/base64", response_model=OCRResponse)
def captcha_base64(data: OCRRequest):
    """解码并识别 Base64 验证码图片。"""

    try:
        image_bytes = decode_base64_image(data.base64_img)
        result = recognize_captcha(image_bytes)
    except ImageTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return OCRResponse(result=result)


def decode_base64_image(value: str) -> bytes:
    """解码纯 Base64 或 data URL，并执行请求大小校验。"""

    payload = "".join((value or "").split())
    if payload.lower().startswith("data:image/"):
        metadata, separator, payload = payload.partition(",")
        if not separator or ";base64" not in metadata.lower():
            raise InvalidImageError("invalid image data URL")

    if not payload:
        raise InvalidImageError("base64_img must not be empty")

    max_encoded_length = ((MAX_IMAGE_BYTES + 2) // 3) * 4
    if len(payload) > max_encoded_length:
        raise ImageTooLargeError("image exceeds the 5 MiB limit")

    padding = len(payload) % 4
    if padding:
        payload += "=" * (4 - padding)

    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError("base64_img is not valid Base64") from exc

    if not image_bytes:
        raise InvalidImageError("decoded image must not be empty")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImageTooLargeError("image exceeds the 5 MiB limit")
    return image_bytes


def preprocess_captcha(image_bytes: bytes) -> bytes:
    """
    生成适合验证码模型识别的深色字符图

    :param image_bytes: 原始验证码图片
    :return: 处理后的 PNG 图片字节
    """
    gray = load_grayscale_image(image_bytes)
    binary = np.where(gray <= DARK_PIXEL_THRESHOLD, 0, 255).astype(np.uint8)

    foreground = (binary == 0).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8,
    )
    keep_labels = stats[:, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA
    keep_labels[0] = False
    cleaned = np.where(keep_labels[labels], 0, 255).astype(np.uint8)

    border = min(BORDER_WIDTH, cleaned.shape[0] // 2, cleaned.shape[1] // 2)
    if border:
        cleaned[:border, :] = 255
        cleaned[-border:, :] = 255
        cleaned[:, :border] = 255
        cleaned[:, -border:] = 255

    encoded, output = cv2.imencode(".png", cleaned)
    if not encoded:
        raise InvalidImageError("failed to preprocess image")
    return output.tobytes()


def build_image_candidates(image_bytes: bytes) -> list[bytes]:
    """
    为彩色遮罩、干扰线和噪点验证码生成多种识别输入

    :param image_bytes: 原始验证码图片
    :return: 去重后的候选 PNG 图片列表
    """
    gray = load_grayscale_image(image_bytes)
    rgb = load_rgb_image(image_bytes)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # 先保留原版本的预处理路径，确保已验证的站点样本不会因新增候选而回归。
    candidates = [preprocess_captcha(image_bytes)]

    # 彩色遮罩会把字符笔画与背景连成一块；仅在检测到明显彩色像素时启用
    # 颜色抑制候选，避免普通验证码每次额外执行模型推理。
    color_mask = (hsv[:, :, 1] > 95) & (hsv[:, :, 2] < 245)
    if float(np.mean(color_mask)) >= 0.01:
        neutral_dark = np.where(
            (hsv[:, :, 1] < 95) & (hsv[:, :, 2] < 210),
            0,
            255,
        ).astype(np.uint8)
        candidates.append(_encode_png(neutral_dark))

    # 保留原图和一个较宽阈值候选，覆盖字符较浅或压缩较重的站点。
    candidates.append(image_bytes)
    candidates.append(_encode_png(np.where(gray <= 100, 0, 255).astype(np.uint8)))

    # 在线条位于前景时提取并相减，再反色还原；直接对白底黑字做开运算
    # 会把白色背景当成待移除对象，反而损坏验证码。
    binary = np.where(gray <= DARK_PIXEL_THRESHOLD, 0, 255).astype(np.uint8)
    foreground = 255 - binary
    for width, height in ((9, 1), (1, 9)):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, height))
        lines = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        without_lines = cv2.subtract(foreground, lines)
        candidates.append(_encode_png(255 - without_lines))

    unique_candidates: list[bytes] = []
    seen: set[bytes] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)
    return unique_candidates


def _encode_png(image: np.ndarray) -> bytes:
    """将灰度矩阵编码为 PNG 字节。"""
    encoded, output = cv2.imencode(".png", image)
    if not encoded:
        raise InvalidImageError("failed to encode candidate image")
    return output.tobytes()


def normalize_prediction(prediction: str) -> str:
    """
    规范化 OCR 输出并过滤模型返回的非验证码字符

    :param prediction: OCR 模型原始结果
    :return: 大写字母数字结果，不符合六位验证码格式时返回空字符串
    """
    result = "".join(re.findall(r"[A-Za-z0-9]", prediction or "")).upper()
    return result if CAPTCHA_PATTERN.fullmatch(result) else ""


def classify_candidate(image_bytes: bytes) -> tuple[str, float]:
    """识别单个候选并返回归一化结果及非空字符置信度。"""
    output = ocr.classification(image_bytes, probability=True)
    charsets = output["charsets"]
    probabilities = output["probability"]
    raw_chars: list[str] = []
    confidence: list[float] = []
    last_index = 0
    for row in probabilities:
        index = max(range(len(row)), key=row.__getitem__)
        if index != last_index and index != 0:
            raw_chars.append(charsets[index])
            confidence.append(float(row[index]))
        last_index = index
    normalized = normalize_prediction("".join(raw_chars))
    return normalized, sum(confidence) / len(confidence) if confidence else 0.0


def load_rgb_image(image_bytes: bytes) -> np.ndarray:
    """校验图片尺寸并转换为不含透明通道的 RGB 矩阵。"""
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            if width <= 0 or height <= 0:
                raise InvalidImageError("image dimensions must be positive")
            if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
                raise ImageTooLargeError("image dimensions exceed 4096 x 4096 pixels")

            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                image = Image.alpha_composite(background, rgba).convert("RGB")
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except ImageTooLargeError:
        raise
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("decoded content is not a supported image") from exc


def load_grayscale_image(image_bytes: bytes) -> np.ndarray:
    """校验图片并转换为灰度矩阵，统一处理透明背景。"""
    return cv2.cvtColor(load_rgb_image(image_bytes), cv2.COLOR_RGB2GRAY)


def recognize_captcha(image_bytes: bytes) -> str:
    """
    使用多候选结果投票识别六位字母数字验证码

    :param image_bytes: 原始验证码图片
    :return: 验证码识别结果，无法确认时返回空字符串
    """
    candidates = build_image_candidates(image_bytes)
    predictions: list[tuple[str, float]] = []
    with ocr_lock:
        primary, primary_confidence = classify_candidate(candidates[0])
        if primary and primary_confidence >= PRIMARY_CONFIDENCE_THRESHOLD:
            return primary
        if primary:
            predictions.append((primary, primary_confidence))

        # 大多数验证码在旧预处理路径上已有较高置信度，只对低置信度或
        # 格式异常结果执行额外推理，避免弱 CPU 主机的响应时间成倍增加。
        for candidate in candidates[1:]:
            normalized, confidence = classify_candidate(candidate)
            if normalized:
                predictions.append((normalized, confidence))

    if not predictions:
        return ""
    if primary:
        # HDSky 的红色遮罩会把首字符 D 误识别为 O；当去干扰候选只修正
        # 这一位且其余字符完全一致时采用 D。其他冲突保留基线结果，避免
        # 阈值候选把清晰的 2/ Z、G/ C 等字符改错。
        corrected = next(
            (
                prediction
                for prediction, _ in predictions
                if prediction[1:] == primary[1:]
                and primary[0] == "O"
                and prediction[0] == "D"
            ),
            None,
        )
        return corrected or primary

    # 先按票数聚合，再用字符置信度打破冲突；置信度可以识别“旧路径只错
    # 一个字符、去线候选却恰好重复”的情况，同时仍保留旧路径的优先级。
    counts: dict[str, int] = {}
    confidence: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for index, (prediction, score) in enumerate(predictions):
        counts[prediction] = counts.get(prediction, 0) + 1
        confidence[prediction] = max(confidence.get(prediction, 0.0), score)
        first_seen.setdefault(prediction, index)
    return max(
        counts,
        key=lambda value: (counts[value], confidence[value], -first_seen[value]),
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9899, reload=False)
