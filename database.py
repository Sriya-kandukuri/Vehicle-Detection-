import sqlite3
from datetime import datetime, timezone
from flask import current_app, g
from werkzeug.security import generate_password_hash


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    phone TEXT,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    stored_name TEXT NOT NULL,
    record_count INTEGER NOT NULL DEFAULT 0,
    class_count INTEGER NOT NULL DEFAULT 0,
    classes TEXT,
    status TEXT NOT NULL DEFAULT 'Uploaded',
    train_count INTEGER NOT NULL DEFAULT 0,
    val_count INTEGER NOT NULL DEFAULT 0,
    test_count INTEGER NOT NULL DEFAULT 0,
    uploaded_by INTEGER,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    algorithm TEXT NOT NULL,
    model_type TEXT NOT NULL,
    dataset_id INTEGER,
    accuracy REAL NOT NULL,
    precision_score REAL NOT NULL,
    recall_score REAL NOT NULL,
    f1_score REAL NOT NULL,
    map50 REAL NOT NULL,
    fps REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'Completed',
    is_active INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    source_type TEXT NOT NULL,
    input_name TEXT NOT NULL,
    output_name TEXT,
    total_vehicles INTEGER NOT NULL DEFAULT 0,
    counts_json TEXT NOT NULL DEFAULT '{}',
    average_confidence REAL NOT NULL DEFAULT 0,
    traffic_level TEXT NOT NULL DEFAULT 'Low',
    processing_seconds REAL NOT NULL DEFAULT 0,
    engine TEXT NOT NULL DEFAULT 'OpenCV',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    message TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'New',
    created_at TEXT NOT NULL
);
"""


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(_error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = get_db()
    db.executescript(SCHEMA)
    dataset_columns = {row["name"] for row in db.execute("PRAGMA table_info(datasets)").fetchall()}
    for column, definition in {
        "dataset_format": "TEXT NOT NULL DEFAULT 'Unknown'",
        "extracted_path": "TEXT",
        "yaml_path": "TEXT",
    }.items():
        if column not in dataset_columns:
            db.execute(f"ALTER TABLE datasets ADD COLUMN {column} {definition}")
    admin = db.execute("SELECT id FROM users WHERE email = ?", ("admin@vehiclevision.ai",)).fetchone()
    if not admin:
        db.execute(
            "INSERT INTO users (name,email,phone,password_hash,role,active,created_at) VALUES (?,?,?,?,?,?,?)",
            ("System Administrator", "admin@vehiclevision.ai", "", generate_password_hash("Admin@123"), "admin", 1, now_iso()),
        )
    if db.execute("SELECT COUNT(*) AS count FROM training_runs").fetchone()["count"] == 0:
        seed_runs = [
            ("YOLOv5", "Existing", 88.4, 87.9, 86.8, 87.3, 89.1, 28.6, 0),
            ("YOLOv8 + EfficientNet + DeepSORT", "Proposed", 94.7, 95.1, 93.8, 94.4, 95.3, 34.8, 1),
        ]
        for algorithm, model_type, accuracy, precision, recall, f1, map50, fps, active in seed_runs:
            db.execute(
                """INSERT INTO training_runs
                (algorithm,model_type,accuracy,precision_score,recall_score,f1_score,map50,fps,status,is_active,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (algorithm, model_type, accuracy, precision, recall, f1, map50, fps, "Demo benchmark", active, now_iso()),
            )
    db.commit()


def init_app(app):
    app.teardown_appcontext(close_db)
    with app.app_context():
        init_db()
