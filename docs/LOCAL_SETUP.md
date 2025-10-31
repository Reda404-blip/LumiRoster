# Local Setup Guide

1. Clone the repository and create a virtual environment.
2. Install Python dependencies with `pip install -r requirements.txt`.
3. Optionally install Node dependencies with `npm install` if you are extending the front-end assets.
4. Duplicate `.env.example` to `.env` and update the Telegram configuration values.
5. Populate `assets/images/students/` with reference photos (JPEG or PNG). Use the student name as the filename to align with database records.
6. Start the FastAPI service: `python -m src.api`.
7. Start the Flask dashboard: `python -m src.main`.
8. Visit `http://127.0.0.1:5000` to access the dashboard. The system automatically creates the SQLite database and cached encodings inside `data/`.

Troubleshooting:
- If `face_recognition` fails to import, ensure that `dlib` is installed. Use a prebuilt wheel on Windows for simplicity.
- Delete the generated files in `data/` if you need to rebuild encodings from updated student images.
