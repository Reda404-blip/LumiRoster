# LumiRoster Vision Suite

## Description
LumiRoster Vision Suite is a computer-vision assisted attendance tracking system. It combines a FastAPI ingestion service, a Flask dashboard with live Socket.IO updates, and asynchronous background processing for camera feeds. The refactor emphasises maintainability and portability so the project is ready for a public GitHub repository and cloud deployment.

## Directory Structure
```
.
├── assets/
│   ├── images/
│   │   └── students/
│   ├── models/
│   ├── static/
│   └── templates/
├── config/
├── data/
│   └── .gitignore
├── docs/
├── logs/
├── src/
│   ├── api.py
│   ├── main.py
│   └── system.py
├── tests/
├── .github/workflows/ci.yml
├── .env.example
├── .gitignore
├── LICENSE
├── package.json
├── package-lock.json
├── requirements.txt
└── README.md
```

- `src/`: Python application sources (FastAPI ingest API, Flask dashboard, async processing helpers).
- `assets/`: Front-end templates, static assets, student reference images, and optional pre-trained models (`assets/models/`).
- `data/`: Runtime database and cached encodings (ignored by Git). Files are generated at runtime.
- `docs/` and `tests/`: Reserved for future documentation and automated tests.
- `.github/workflows/ci.yml`: Continuous integration pipeline definition.

## Installation
### Prerequisites
- Python 3.11+
- Node.js 20+ (optional, required only if you extend the front-end build pipeline)
- CMake build tools if you plan to compile `dlib` locally, otherwise use a prebuilt wheel.

### Python setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Node.js setup (optional)
```bash
npm install
```

## Configuration
Copy `.env.example` to `.env` and provide the required secrets:
```
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
TELEGRAM_USER_ID=telegram-user-or-chat-id
```

Additional configuration values live in `config/config.yaml`. Update paths or thresholds as required for your environment.

Place student reference photos inside `assets/images/students/` and store large model files (for example dlib shape predictors) in `assets/models/`. Cached encodings and the SQLite database are written to `data/` at runtime.

## Usage
### Run the FastAPI ingestion service
```bash
python -m src.api
```

### Run the Flask dashboard with live recognition loop
```bash
python -m src.main
```
This launches the Socket.IO-enabled Flask server and begins webcam detection. Press `q` in the video window to stop recognition.

### Processing frames asynchronously
`src/system.py` exposes the `AttendanceSystem` class for integrating additional camera feeds or expanding analytics. Instantiate it within your own orchestration code to process frames and send notifications.

## Deployment
- **GitHub Pages**: Static assets in `assets/static/` can be published with a dedicated build step. Export dashboards to static HTML if you intend to host without Flask.
- **Heroku / Render / Railway**: Use the supplied `requirements.txt` and configure the Procfile of your choice (e.g., `web: uvicorn src.api:app --host=0.0.0.0 --port=${PORT}`). Ensure environment variables are set in the platform dashboard.
- **Vercel / Netlify**: Run the FastAPI app behind a serverless adapter or host the Flask application on services that support Python web servers.

## Continuous Integration
Every push or pull request against `main` or `dev` runs the GitHub Actions workflow in `.github/workflows/ci.yml`. The pipeline installs Python and Node dependencies, performs a static compilation check, and optionally triggers the Node build script.

## Versioning & Releases
- Adopt semantic versioning (`v1.0.0`).
- Tag major release candidates via `git tag v1.0.0 && git push origin v1.0.0`.
- Maintain a changelog in `docs/` (e.g., `docs/CHANGELOG.md`) as the project evolves.

## Contributing
1. Fork the repository and clone locally.
2. Create a feature branch from `dev` (e.g., `feature/camera-api`).
3. Follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages (`feat: add websocket summary`).
4. Open a pull request against `dev`; squash merges into `main` when ready to release.

### Branching Strategy
- `main`: production-ready, tagged releases only.
- `dev`: integration branch for completed features prior to release.
- Short-lived branches (`feature/*`, `bugfix/*`, `chore/*`) for focused work.

### Code Style & Testing
- Run `python -m compileall src` before opening a pull request.
- Add unit tests under `tests/` for new behaviour.
- Keep configuration and secrets out of version control; rely on environment variables.

## License
Released under the [MIT License](LICENSE).

## Acknowledgements
This project builds on the `face_recognition` and OpenCV ecosystems. Be sure to comply with their respective licenses when distributing binaries.

## Maintainer
Built and maintained by Reda El Maaroufi.
