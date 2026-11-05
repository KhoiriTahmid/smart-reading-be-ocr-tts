"""RapidOCR wrapper (PP-OCRv6 tiny, ONNX Runtime on CPU)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrResult:
    text: str
    lines: int
    avg_confidence: float


class OcrEngine:
    """Loaded once at application startup and reused for every request."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine = None

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def load(self) -> None:
        if self._engine is not None:
            return

        os.environ["OMP_NUM_THREADS"] = str(self._settings.omp_num_threads)

        # Imported lazily so unit tests that use a fake OCR engine do not need
        # to import onnxruntime or the model package.
        from rapidocr import ModelType, OCRVersion, RapidOCR

        params: dict[str, object] = {
            "Det.ocr_version": OCRVersion.PPOCRV6,
            "Rec.ocr_version": OCRVersion.PPOCRV6,
            "Det.model_type": ModelType.TINY,
            "Rec.model_type": ModelType.TINY,
            # EXIF orientation is already applied by app.utils.images, so the
            # 180-degree classifier would only add another model and latency.
            "Global.use_cls": False,
            "Global.text_score": self._settings.ocr_min_confidence,
            "Global.max_side_len": self._settings.ocr_max_side,
            "EngineConfig.onnxruntime.intra_op_num_threads": (
                self._settings.omp_num_threads
            ),
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
        if self._settings.ocr_model_dir is not None:
            self._settings.ocr_model_dir.mkdir(parents=True, exist_ok=True)
            params["Global.model_root_dir"] = str(self._settings.ocr_model_dir)

        self._engine = RapidOCR(params=params)
        logger.info("OCR engine initialised (PP-OCRv6 tiny, ONNX Runtime CPU)")

    def warmup(self) -> None:
        """Force detector and recogniser sessions to be created.

        Used by ``scripts/warmup_models.py`` during the Docker image build so
        no model is downloaded on the first real request.
        """

        self.load()
        assert self._engine is not None
        for method_name in ("_load_det_model", "_load_rec_model"):
            method = getattr(self._engine, method_name, None)
            if callable(method):
                method()

    def extract(self, image_rgb: np.ndarray) -> OcrResult:
        """Run OCR on an RGB uint8 array and post-process the result."""

        if self._engine is None:
            raise RuntimeError("OCR engine is not loaded")

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("OCR input must be an RGB image")

        # RapidOCR expects BGR when given a numpy array (it only does the
        # RGB->BGR conversion itself for bytes/paths/PIL images).
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        result = self._engine(image_bgr)

        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        if texts is None or len(texts) == 0:
            return OcrResult(text="", lines=0, avg_confidence=0.0)

        kept: list[tuple[str, float]] = []
        score_list = list(scores) if scores is not None else []
        for index, raw_text in enumerate(texts):
            if not isinstance(raw_text, str):
                continue
            text = raw_text.strip()
            if not text:
                continue

            confidence = 0.0
            if index < len(score_list):
                try:
                    confidence = float(score_list[index])
                except (TypeError, ValueError):
                    confidence = 0.0

            if confidence < self._settings.ocr_min_confidence:
                continue
            kept.append((text, confidence))

        if not kept:
            return OcrResult(text="", lines=0, avg_confidence=0.0)

        joined = "\n".join(text for text, _ in kept).strip()
        average = round(sum(score for _, score in kept) / len(kept), 4)
        return OcrResult(text=joined, lines=len(kept), avg_confidence=average)
