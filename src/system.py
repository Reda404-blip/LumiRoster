import asyncio
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List

import aiohttp
import cv2
import face_recognition
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = PROJECT_ROOT / "assets" / "images" / "students"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
NOTIFICATION_COOLDOWN = timedelta(minutes=5)


@dataclass(frozen=True)
class DetectionResult:
    name: str
    confidence: float
    camera_id: str
    timestamp: datetime

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "confidence": self.confidence,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp.isoformat(),
        }


class AttendanceSystem:
    def __init__(self, database_path: str | Path | None = None, image_dir: str | Path | None = None):
        self.db_path = Path(database_path) if database_path else DATA_DIR / "attendance.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.image_dir = Path(image_dir) if image_dir else IMAGES_DIR
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.notified_students: Dict[str, datetime] = {}
        self.logger = self._configure_logger()
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.telegram_user_id = os.getenv("TELEGRAM_USER_ID", "").strip()
        self.notification_cooldown = NOTIFICATION_COOLDOWN
        self._setup_database()

    def _configure_logger(self) -> logging.Logger:
        logger = logging.getLogger("attendance_system")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(LOGS_DIR / "attendance_system.log")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
        return logger

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _setup_database(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    email TEXT,
                    student_id TEXT UNIQUE,
                    image_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    student_id INTEGER,
                    status TEXT NOT NULL,
                    camera_id TEXT,
                    confidence_score REAL,
                    FOREIGN KEY (student_id) REFERENCES students(id)
                );
                """
            )
            connection.commit()

    def load_reference_faces(self) -> None:
        if not self.image_dir.is_dir():
            self.logger.warning("Student image directory %s not found.", self.image_dir)
            return

        self.known_face_encodings.clear()
        self.known_face_names.clear()

        with self._connect() as connection:
            cursor = connection.cursor()
            for image_path in self._iter_image_files():
                student_name = image_path.stem
                cursor.execute(
                    "INSERT OR IGNORE INTO students (name, image_path) VALUES (?, ?)",
                    (student_name, str(image_path)),
                )

                image = cv2.imread(str(image_path))
                if image is None or image.ndim != 3 or image.shape[2] != 3:
                    self.logger.warning("Skipping %s: unable to read RGB data.", image_path)
                    continue

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_image)
                if not encodings:
                    self.logger.info("No face detected in %s; skipping.", image_path.name)
                    continue

                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(student_name)

            connection.commit()

        self.logger.info("Loaded %d reference faces.", len(self.known_face_names))

    def _iter_image_files(self) -> Iterable[Path]:
        return sorted(
            path
            for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    async def process_frame(self, frame: np.ndarray, camera_id: str) -> Dict[str, object]:
        detections = await asyncio.to_thread(self._recognize_faces, frame, camera_id)
        payload = {"frame_id": id(frame), "results": []}

        for detection in detections:
            payload["results"].append(detection.as_dict())
            if self._should_notify(detection):
                await self._log_and_notify(detection)

        return payload

    def _recognize_faces(self, frame: np.ndarray, camera_id: str) -> List[DetectionResult]:
        try:
            scaled_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)

            locations = face_recognition.face_locations(rgb_frame)
            encodings = face_recognition.face_encodings(rgb_frame, locations)

            detections: List[DetectionResult] = []
            for encoding in encodings:
                match = self._match_face(encoding)
                if match is None:
                    continue
                name, confidence = match
                detections.append(
                    DetectionResult(
                        name=name,
                        confidence=confidence,
                        camera_id=camera_id,
                        timestamp=datetime.now(),
                    )
                )
            return detections
        except Exception as exc:
            self.logger.error("Error processing frame: %s", exc)
            return []

    def _match_face(self, face_encoding: np.ndarray) -> tuple[str, float] | None:
        if not self.known_face_encodings:
            return None

        distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        best_index = int(np.argmin(distances))
        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)

        if not matches[best_index]:
            return None

        confidence = float(max(0.0, 1.0 - distances[best_index]))
        return self.known_face_names[best_index], confidence

    def _should_notify(self, detection: DetectionResult) -> bool:
        last_notification = self.notified_students.get(detection.name)
        if last_notification and detection.timestamp - last_notification < self.notification_cooldown:
            return False
        self.notified_students[detection.name] = detection.timestamp
        return True

    async def _log_and_notify(self, detection: DetectionResult) -> None:
        await asyncio.to_thread(self._log_attendance, detection)
        await self._send_notification(detection)

    def _log_attendance(self, detection: DetectionResult) -> None:
        with self._connect() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT id FROM students WHERE name = ?", (detection.name,))
            row = cursor.fetchone()
            student_id = row["id"] if row else None

            cursor.execute(
                """
                INSERT INTO attendance (date, student_id, status, camera_id, confidence_score)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    detection.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    student_id,
                    "Present",
                    detection.camera_id,
                    detection.confidence,
                ),
            )
            connection.commit()

    async def _send_notification(self, detection: DetectionResult) -> None:
        message = (
            f"Attendance Logged\n"
            f"Student: {detection.name}\n"
            f"Time: {detection.timestamp:%Y-%m-%d %H:%M:%S}\n"
            f"Confidence: {detection.confidence:.2f}\n"
            f"Camera: {detection.camera_id}"
        )
        await self._send_telegram(message)

    async def _send_telegram(self, message: str) -> None:
        if not self.telegram_token or not self.telegram_user_id:
            self.logger.debug("Telegram credentials missing; skipping notification.")
            return

        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data={"chat_id": self.telegram_user_id, "text": message}) as response:
                    if response.status != 200:
                        self.logger.warning("Telegram API error: %s", await response.text())
        except Exception as exc:
            self.logger.error("Failed to send Telegram message: %s", exc)
