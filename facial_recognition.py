"""Simple facial recognition pipeline built on OpenCV's LBPH recognizer.

This module provides a small helper around OpenCV to train a recognizer from a
folder-based dataset and run predictions on new images. It intentionally keeps
state in memory so it can be used from scripts or notebooks without any
additional infrastructure.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Recognition:
    """Face recognition prediction.

    Attributes:
        label: The assigned class label.
        confidence: Distance score from the recognizer (lower is better).
        box: Bounding box of the detected face in ``(x, y, w, h)`` format.
    """

    label: str
    confidence: float
    box: Tuple[int, int, int, int]


class FaceRecognizer:
    """Train and apply an LBPH-based facial recognition model.

    Example
    -------
    >>> recognizer = FaceRecognizer()
    >>> recognizer.train("./dataset")
    >>> recognitions = recognizer.recognize("./photo.jpg")
    """

    def __init__(self, cascade_path: Optional[str] = None) -> None:
        cascade_dir = Path(cv2.data.haarcascades)
        default_cascade = cascade_dir / "haarcascade_frontalface_default.xml"
        chosen = Path(cascade_path) if cascade_path is not None else default_cascade
        if not chosen.exists():
            raise FileNotFoundError(
                f"Face cascade file not found at {chosen}. "
                "Pass a valid path via cascade_path."
            )

        self._detector = cv2.CascadeClassifier(str(chosen))
        self._recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._label_to_idx: Dict[str, int] = {}
        self._idx_to_label: Dict[int, str] = {}
        self._is_trained = False

    @staticmethod
    def _normalize_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _detect_faces(self, gray_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        faces = self._detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

    def _prepare_dataset(self, dataset_dir: Path) -> Tuple[List[np.ndarray], List[int]]:
        images: List[np.ndarray] = []
        labels: List[int] = []

        for label_idx, label_dir in enumerate(sorted(dataset_dir.iterdir())):
            if not label_dir.is_dir():
                continue

            label = label_dir.name
            self._label_to_idx[label] = label_idx
            self._idx_to_label[label_idx] = label

            for image_path in sorted(label_dir.glob("*")):
                if not image_path.is_file():
                    continue
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                gray = self._normalize_gray(image)
                faces = self._detect_faces(gray)
                for (x, y, w, h) in faces:
                    face_region = gray[y : y + h, x : x + w]
                    images.append(face_region)
                    labels.append(label_idx)
        if not images:
            raise ValueError(
                "No valid face images found. Ensure the dataset is organized as "
                "<dataset>/<person_name>/*.jpg and faces are detectable."
            )
        return images, labels

    def train(self, dataset_dir: str | Path) -> None:
        """Train the recognizer using images grouped by label directories."""

        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory {dataset_path} not found.")

        images, labels = self._prepare_dataset(dataset_path)
        self._recognizer.train(images, np.array(labels))
        self._is_trained = True

    def recognize(self, image_path: str | Path) -> List[Recognition]:
        """Run face detection and recognition on a single image."""

        if not self._is_trained:
            raise RuntimeError("The recognizer has not been trained yet. Call train() first.")

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image at {image_path}.")

        gray = self._normalize_gray(image)
        boxes = self._detect_faces(gray)
        results: List[Recognition] = []
        for (x, y, w, h) in boxes:
            face_region = gray[y : y + h, x : x + w]
            predicted_idx, confidence = self._recognizer.predict(face_region)
            label = self._idx_to_label.get(predicted_idx, "unknown")
            results.append(Recognition(label=label, confidence=confidence, box=(x, y, w, h)))
        return results

    def annotate(self, image_path: str | Path, output_path: str | Path) -> List[Recognition]:
        """Recognize faces in an image and save an annotated copy."""

        recognitions = self.recognize(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image at {image_path}.")

        for rec in recognitions:
            x, y, w, h = rec.box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label_text = f"{rec.label} ({rec.confidence:.1f})"
            cv2.putText(
                image,
                label_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        return recognitions


__all__ = ["Recognition", "FaceRecognizer"]
