import csv
import logging
import os
import sqlite3
from datetime import datetime, time, timedelta
from io import BytesIO, StringIO
from pathlib import Path
from threading import Thread
from typing import Iterable, List, Sequence, Tuple

import cv2
import face_recognition
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from flask_socketio import SocketIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = PROJECT_ROOT / "assets"
TEMPLATES_DIR = ASSETS_DIR / "templates"
STATIC_DIR = ASSETS_DIR / "static"
STUDENTS_IMAGE_DIR = ASSETS_DIR / "images" / "students"
STUDENTS_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_START_GRACE_MINUTES = 5
ATTENDANCE_DB = DATA_DIR / "attendance.db"
ENCODINGS_FILE = DATA_DIR / "face_encodings.npy"
NAMES_FILE = DATA_DIR / "face_names.npy"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_USER_ID = os.getenv("TELEGRAM_USER_ID", "").strip()

app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
)
socketio = SocketIO(app)

class_start_time: time | None = None


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(ATTENDANCE_DB)
    connection.row_factory = sqlite3.Row
    return connection


def create_database(clear_attendance: bool = True) -> None:
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                student_name TEXT NOT NULL,
                status TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                image_path TEXT NOT NULL
            )
            """
        )
        if clear_attendance:
            logger.info("Clearing existing attendance records.")
            cursor.execute("DELETE FROM attendance")
        connection.commit()


def set_class_start_time(start_hour: int, start_minute: int) -> None:
    global class_start_time
    class_start_time = time(hour=start_hour, minute=start_minute)
    logger.info("Class start time set to %s", class_start_time.strftime("%H:%M"))


def _attendance_status(check_in_time: datetime) -> str:
    if class_start_time is None:
        return "Present"
    start_datetime = datetime.combine(check_in_time.date(), class_start_time)
    grace_deadline = start_datetime + timedelta(minutes=CLASS_START_GRACE_MINUTES)
    return "Present" if check_in_time <= grace_deadline else "Late"


def send_telegram_message(message: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_USER_ID:
        logger.debug("Skipping Telegram message; credentials are not configured.")
        return

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_USER_ID, "text": message},
            timeout=10,
        )
        response.raise_for_status()
        logger.info("Telegram notification sent.")
    except requests.RequestException as exc:
        logger.warning("Failed to send Telegram message: %s", exc)


def log_attendance(student_name: str) -> None:
    timestamp = datetime.now()
    status = _attendance_status(timestamp)

    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO attendance (date, student_name, status) VALUES (?, ?, ?)",
            (timestamp.strftime("%Y-%m-%d %H:%M:%S"), student_name, status),
        )
        connection.commit()

    message = (
        f"Attendance logged: {student_name} is {status} "
        f"at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    send_telegram_message(message)

    socketio.emit(
        "attendance_update",
        {
            "type": "attendance_update",
            "student_name": student_name,
            "status": status,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def _iter_image_files(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()
    )


def load_student_faces(
    image_directory: Path = STUDENTS_IMAGE_DIR,
    encoding_file: Path = ENCODINGS_FILE,
    names_file: Path = NAMES_FILE,
) -> Tuple[List[np.ndarray], List[str]]:
    image_directory = Path(image_directory)
    encoding_file = Path(encoding_file)
    names_file = Path(names_file)

    if encoding_file.is_file() and names_file.is_file():
        logger.info("Loading cached face encodings.")
        encodings = np.load(encoding_file, allow_pickle=True).tolist()
        names = np.load(names_file, allow_pickle=True).tolist()
        return encodings, names

    known_face_encodings: List[np.ndarray] = []
    known_face_names: List[str] = []

    with get_connection() as connection:
        cursor = connection.cursor()
        for image_path in _iter_image_files(image_directory):
            student_name = image_path.stem
            cursor.execute(
                "INSERT OR IGNORE INTO students (name, image_path) VALUES (?, ?)",
                (student_name, str(image_path)),
            )

            logger.info("Processing reference image %s", image_path.name)
            image = cv2.imread(str(image_path))
            if image is None or image.ndim != 3 or image.shape[2] != 3:
                logger.warning("Skipping %s: invalid image data.", image_path)
                continue

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_image)
            if not encodings:
                logger.warning("No detectable face in %s; skipping.", image_path)
                continue

            known_face_encodings.append(encodings[0])
            known_face_names.append(student_name)

        connection.commit()

    np.save(encoding_file, np.array(known_face_encodings, dtype=object))
    np.save(names_file, np.array(known_face_names, dtype=object))
    logger.info("Stored %d face encodings for future runs.", len(known_face_names))
    return known_face_encodings, known_face_names


def run_detection_loop(
    video_capture: cv2.VideoCapture,
    known_face_encodings: Sequence[np.ndarray],
    known_face_names: Sequence[str],
) -> None:
    if not known_face_encodings:
        logger.warning("No reference faces available; detection loop will not log attendance.")

    process_frame = True
    recorded_students = set()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            logger.error("Failed to capture frame from camera.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        if process_frame and known_face_encodings:
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                if True not in matches:
                    continue

                name = known_face_names[matches.index(True)]
                if name in recorded_students:
                    continue

                recorded_students.add(name)
                logger.info("Marking %s as present.", name)
                log_attendance(name)

        process_frame = not process_frame

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


@app.route("/")
def dashboard():
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT student_name, status, date "
            "FROM attendance ORDER BY date DESC LIMIT 10"
        ).fetchall()
    return render_template("dashboard.html", attendance_records=rows)


@app.route("/set_start_time", methods=["POST"])
def set_start_time():
    data = request.get_json(force=True, silent=True) or {}
    start_time_str = data.get("start_time")
    if not start_time_str:
        return jsonify({"success": False, "error": "Missing start_time field"}), 400

    try:
        parsed_time = datetime.strptime(start_time_str, "%H:%M")
    except ValueError:
        return jsonify({"success": False, "error": "Invalid time format"}), 400

    set_class_start_time(parsed_time.hour, parsed_time.minute)
    return jsonify({"success": True, "message": "Class start time updated successfully"})


@app.route("/students")
def students():
    with get_connection() as connection:
        students_data = connection.execute(
            "SELECT name, image_path FROM students ORDER BY name"
        ).fetchall()
    return render_template("students.html", students=students_data)


def _serialize_attendance(rows) -> List[dict]:
    return [
        {
            "student_name": row["student_name"],
            "status": row["status"],
            "timestamp": row["date"],
        }
        for row in rows
    ]


@app.route("/api/attendance")
def get_attendance():
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT student_name, status, date FROM attendance ORDER BY date DESC"
        ).fetchall()
    return jsonify(_serialize_attendance(rows))


@app.route("/api/search")
def search_attendance():
    search_term = request.args.get("term", "").strip()
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT student_name, status, date FROM attendance "
            "WHERE student_name LIKE ? ORDER BY date DESC",
            (f"%{search_term}%",),
        ).fetchall()
    return jsonify(_serialize_attendance(rows))


@app.route("/api/export")
def export_attendance():
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT student_name, status, date FROM attendance ORDER BY date DESC"
        ).fetchall()

    output = StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(["Student Name", "Status", "Timestamp"])
    writer.writerows((row["student_name"], row["status"], row["date"]) for row in rows)

    buffer = BytesIO(output.getvalue().encode("utf-8"))
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name="attendance.csv",
    )


@app.route("/students_images/<path:filename>")
def serve_student_image(filename: str):
    return send_from_directory(str(STUDENTS_IMAGE_DIR), filename)


@socketio.on("connect")
def handle_connect():
    logger.info("Client connected.")


@socketio.on("disconnect")
def handle_disconnect():
    logger.info("Client disconnected.")


def main(reset_attendance: bool = False) -> None:
    create_database(clear_attendance=reset_attendance)
    set_class_start_time(9, 0)
    encodings, names = load_student_faces()

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        logger.error("Unable to access webcam.")
        return

    try:
        run_detection_loop(video_capture, encodings, names)
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    server_thread = Thread(
        target=lambda: socketio.run(app, debug=True, use_reloader=False),
        name="FlaskThread",
        daemon=True,
    )
    server_thread.start()
    try:
        main(reset_attendance=True)
    except KeyboardInterrupt:
        logger.info("Shutting down attendance system.")
