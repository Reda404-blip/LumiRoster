import asyncio
import logging
import socket
import sqlite3
from datetime import date, datetime, time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "attendance.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

lesson_start_time: Optional[time] = None


class DetectionData(BaseModel):
    timestamp: str
    detections: List[dict]
    frame: str
    totalStudents: int
    studentName: Optional[str]
    action: str
    attendanceData: Dict[str, List[str]]


class LessonStartTime(BaseModel):
    start_time: str


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_database() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                status TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.commit()


def log_attendance(student_name: str, status: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO attendance (student_name, status, timestamp) VALUES (?, ?, ?)",
            (student_name, status, datetime.now().isoformat()),
        )
        conn.commit()
    logger.info("Logged %s as %s via API ingestion.", student_name, status)


@app.post("/api/v1/detections")
async def receive_detection_data(detection_data: DetectionData):
    logger.debug("Incoming detection payload: %s", detection_data.json())
    if detection_data.studentName and detection_data.action:
        log_attendance(detection_data.studentName, detection_data.action)
    return {"status": "success"}


@app.get("/attendance/today")
async def get_today_attendance():
    today = date.today().isoformat()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM attendance WHERE date(timestamp) = ? ORDER BY timestamp DESC",
            (today,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "student_name": row["student_name"],
            "status": row["status"],
            "timestamp": row["timestamp"],
        }
        for row in rows
    ]


@app.post("/lesson/start-time")
async def set_lesson_start_time(start_time: LessonStartTime):
    global lesson_start_time
    try:
        lesson_start_time = datetime.strptime(start_time.start_time, "%H:%M").time()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid time format. Use HH:MM") from exc
    logger.info("Lesson start time updated to %s", lesson_start_time)
    return {"status": "success", "start_time": lesson_start_time.strftime("%H:%M")}


@app.get("/lesson/start-time")
async def get_lesson_start_time():
    return {"start_time": lesson_start_time.strftime("%H:%M") if lesson_start_time else None}


async def _stream_attendance_summary(websocket: WebSocket):
    while True:
        with get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
            latest = conn.execute(
                "SELECT student_name, status, timestamp FROM attendance "
                "ORDER BY timestamp DESC LIMIT 5"
            ).fetchall()

        payload = {
            "totalStudents": total,
            "recent": [
                {
                    "studentName": row["student_name"],
                    "status": row["status"],
                    "timestamp": row["timestamp"],
                }
                for row in latest
            ],
            "lessonStart": lesson_start_time.strftime("%H:%M") if lesson_start_time else None,
        }
        await websocket.send_json(payload)
        await asyncio.sleep(5)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected.")
    try:
        await _stream_attendance_summary(websocket)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")


def find_available_port(start_port: int = 8000, max_port: int = 9000) -> int:
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError("No free ports in range.")


if __name__ == "__main__":
    import uvicorn

    create_database()
    port = find_available_port()
    logger.info("Starting FastAPI server on port %d", port)
    uvicorn.run(app, host="127.0.0.1", port=port)
