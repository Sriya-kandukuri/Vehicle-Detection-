import base64
import json
import secrets
from collections import Counter
from functools import wraps
from pathlib import Path

from flask import (
    Flask, Response, abort, flash, jsonify, redirect, render_template, request,
    send_file, send_from_directory, session, url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from config import Config
from database import get_db, init_app as init_database, now_iso
from reporting import detections_csv, detections_pdf
from vehicle_ai import VehicleDetector, benchmark_metrics, extract_dataset_archive, inspect_dataset


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)
    for key in ("UPLOAD_FOLDER", "OUTPUT_FOLDER", "DATASET_FOLDER"):
        Path(app.config[key]).mkdir(parents=True, exist_ok=True)
    Path(app.config["DATABASE"]).parent.mkdir(parents=True, exist_ok=True)
    init_database(app)

    detector_holder = {}

    def detector():
        if "engine" not in detector_holder:
            detector_holder["engine"] = VehicleDetector()
        return detector_holder["engine"]

    def login_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if not session.get("user_id"):
                flash("Please log in to continue.", "warning")
                return redirect(url_for("login"))
            return view(*args, **kwargs)
        return wrapped

    def admin_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if session.get("role") != "admin":
                flash("Administrator access is required.", "danger")
                return redirect(url_for("login"))
            return view(*args, **kwargs)
        return wrapped

    @app.context_processor
    def inject_globals():
        return {
            "current_year": 2026,
            "session_user": session.get("name"),
            "max_upload_mb": app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
        }

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip().lower()
            phone = request.form.get("phone", "").strip()
            password = request.form.get("password", "")
            if len(name) < 2 or "@" not in email or len(password) < 8:
                flash("Enter a valid name, email, and a password of at least 8 characters.", "danger")
                return render_template("register.html")
            db = get_db()
            try:
                db.execute(
                    "INSERT INTO users (name,email,phone,password_hash,role,active,created_at) VALUES (?,?,?,?,?,?,?)",
                    (name, email, phone, generate_password_hash(password), "user", 1, now_iso()),
                )
                db.commit()
            except Exception:
                flash("An account with that email already exists.", "danger")
                return render_template("register.html")
            flash("Registration completed. You can now log in.", "success")
            return redirect(url_for("login"))
        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            user = get_db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            if not user or not check_password_hash(user["password_hash"], request.form.get("password", "")):
                flash("Invalid email address or password.", "danger")
            elif not user["active"]:
                flash("This account is inactive. Contact the administrator.", "warning")
            else:
                session.clear()
                session.update(user_id=user["id"], name=user["name"], role=user["role"])
                flash(f'Welcome back, {user["name"]}.', "success")
                return redirect(url_for("admin_dashboard" if user["role"] == "admin" else "user_dashboard"))
        return render_template("login.html")

    @app.get("/logout")
    def logout():
        session.clear()
        flash("You have been logged out.", "info")
        return redirect(url_for("index"))

    @app.get("/dashboard")
    @login_required
    def user_dashboard():
        if session.get("role") == "admin":
            return redirect(url_for("admin_dashboard"))
        db = get_db()
        stats = db.execute(
            """SELECT COUNT(*) sessions, COALESCE(SUM(total_vehicles),0) vehicles,
            COALESCE(AVG(average_confidence),0) confidence FROM detections WHERE user_id=?""",
            (session["user_id"],),
        ).fetchone()
        recent = db.execute("SELECT * FROM detections WHERE user_id=? ORDER BY id DESC LIMIT 5", (session["user_id"],)).fetchall()
        all_rows = db.execute("SELECT counts_json FROM detections WHERE user_id=?", (session["user_id"],)).fetchall()
        class_counts = Counter()
        for row in all_rows:
            class_counts.update(json.loads(row["counts_json"] or "{}"))
        return render_template("user/dashboard.html", stats=stats, recent=recent, class_counts=dict(class_counts))

    @app.route("/detect", methods=["GET", "POST"])
    @login_required
    def detect():
        if request.method == "POST":
            uploaded = request.files.get("media")
            if not uploaded or not uploaded.filename:
                flash("Choose an image or video file.", "danger")
                return redirect(url_for("detect"))
            original = secure_filename(uploaded.filename)
            extension = Path(original).suffix.lower().lstrip(".")
            allowed_images = app.config["ALLOWED_IMAGE_EXTENSIONS"]
            allowed_videos = app.config["ALLOWED_VIDEO_EXTENSIONS"]
            if extension not in allowed_images | allowed_videos:
                flash("Unsupported file type. Upload JPG, PNG, MP4, AVI, MOV, MKV or WEBM.", "danger")
                return redirect(url_for("detect"))
            token = secrets.token_hex(8)
            stored = f"{token}_{original}"
            input_path = Path(app.config["UPLOAD_FOLDER"]) / stored
            source_type = "Image" if extension in allowed_images else "Video"
            output_extension = ".jpg" if source_type == "Image" else ".mp4"
            output_name = f"detected_{token}{output_extension}"
            output_path = Path(app.config["OUTPUT_FOLDER"]) / output_name
            uploaded.save(input_path)
            try:
                if source_type == "Image":
                    summary = detector().process_image(input_path, output_path)
                else:
                    summary = detector().process_video(input_path, output_path)
            except Exception as exc:
                input_path.unlink(missing_ok=True)
                flash(f"Detection failed: {exc}", "danger")
                return redirect(url_for("detect"))
            db = get_db()
            cursor = db.execute(
                """INSERT INTO detections
                (user_id,source_type,input_name,output_name,total_vehicles,counts_json,average_confidence,
                traffic_level,processing_seconds,engine,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (session["user_id"], source_type, original, output_name, summary["total"], json.dumps(summary["counts"]),
                 summary["average_confidence"], summary["traffic_level"], summary["processing_seconds"], summary["engine"], now_iso()),
            )
            db.commit()
            return redirect(url_for("detection_result", detection_id=cursor.lastrowid))
        return render_template("user/detect.html")

    @app.get("/live")
    @login_required
    def live_detection():
        return render_template("user/live.html")

    @app.post("/api/detect-frame")
    @login_required
    def detect_frame_api():
        frame = request.files.get("frame")
        if not frame:
            return jsonify(error="Missing camera frame"), 400
        try:
            encoded, summary = detector().process_bytes(frame.read())
            return jsonify(image="data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"), summary=summary)
        except Exception as exc:
            return jsonify(error=str(exc)), 400

    @app.post("/live/save")
    @login_required
    def save_live_session():
        payload = request.get_json(silent=True) or {}
        counts = payload.get("counts", {})
        if not isinstance(counts, dict):
            return jsonify(error="Invalid counts"), 400
        total = int(sum(max(0, int(value)) for value in counts.values()))
        cursor = get_db().execute(
            """INSERT INTO detections
            (user_id,source_type,input_name,output_name,total_vehicles,counts_json,average_confidence,
            traffic_level,processing_seconds,engine,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (session["user_id"], "Live Camera", "Browser camera session", "", total, json.dumps(counts),
             float(payload.get("confidence", 0)), "High" if total >= 15 else "Medium" if total >= 6 else "Low",
             float(payload.get("seconds", 0)), detector().engine, now_iso()),
        )
        get_db().commit()
        return jsonify(ok=True, redirect=url_for("detection_result", detection_id=cursor.lastrowid))

    @app.get("/result/<int:detection_id>")
    @login_required
    def detection_result(detection_id):
        row = get_db().execute("SELECT * FROM detections WHERE id=?", (detection_id,)).fetchone()
        if not row or (session.get("role") != "admin" and row["user_id"] != session["user_id"]):
            abort(404)
        return render_template("user/result.html", detection=row, counts=json.loads(row["counts_json"] or "{}"))

    @app.get("/history")
    @login_required
    def history():
        rows = get_db().execute("SELECT * FROM detections WHERE user_id=? ORDER BY id DESC", (session["user_id"],)).fetchall()
        return render_template("user/history.html", rows=rows)

    @app.post("/history/<int:detection_id>/delete")
    @login_required
    def delete_history(detection_id):
        db = get_db()
        row = db.execute("SELECT * FROM detections WHERE id=? AND user_id=?", (detection_id, session["user_id"])).fetchone()
        if row:
            if row["output_name"]:
                (Path(app.config["OUTPUT_FOLDER"]) / row["output_name"]).unlink(missing_ok=True)
            db.execute("DELETE FROM detections WHERE id=?", (detection_id,))
            db.commit()
            flash("Detection record deleted.", "success")
        return redirect(url_for("history"))

    @app.post("/feedback")
    @login_required
    def feedback():
        message = request.form.get("message", "").strip()
        if len(message) >= 5:
            db = get_db()
            db.execute("INSERT INTO feedback (user_id,message,status,created_at) VALUES (?,?,?,?)", (session["user_id"], message, "New", now_iso()))
            db.commit()
            flash("Thank you. Your feedback was submitted.", "success")
        return redirect(request.referrer or url_for("user_dashboard"))

    @app.get("/media/output/<path:filename>")
    @login_required
    def output_media(filename):
        return send_from_directory(app.config["OUTPUT_FOLDER"], secure_filename(filename))

    @app.get("/admin")
    @admin_required
    def admin_dashboard():
        db = get_db()
        stats = {
            "users": db.execute("SELECT COUNT(*) count FROM users WHERE role='user'").fetchone()["count"],
            "detections": db.execute("SELECT COUNT(*) count FROM detections").fetchone()["count"],
            "vehicles": db.execute("SELECT COALESCE(SUM(total_vehicles),0) count FROM detections").fetchone()["count"],
            "datasets": db.execute("SELECT COUNT(*) count FROM datasets").fetchone()["count"],
        }
        recent = db.execute("""SELECT d.*,u.name user_name FROM detections d JOIN users u ON u.id=d.user_id ORDER BY d.id DESC LIMIT 7""").fetchall()
        models = db.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT 4").fetchall()
        return render_template("admin/dashboard.html", stats=stats, recent=recent, models=models)

    @app.route("/admin/datasets", methods=["GET", "POST"])
    @admin_required
    def admin_datasets():
        db = get_db()
        if request.method == "POST":
            uploaded = request.files.get("dataset")
            if not uploaded or not uploaded.filename:
                flash("Choose a CSV or ZIP dataset.", "danger")
                return redirect(url_for("admin_datasets"))
            original = secure_filename(uploaded.filename)
            if Path(original).suffix.lower() not in {".csv", ".zip"}:
                flash("Dataset must be a CSV file or a ZIP containing class folders.", "danger")
                return redirect(url_for("admin_datasets"))
            stored = f"{secrets.token_hex(7)}_{original}"
            path = Path(app.config["DATASET_FOLDER"]) / stored
            uploaded.save(path)
            try:
                details = inspect_dataset(path)
            except Exception as exc:
                path.unlink(missing_ok=True)
                flash(f"Dataset validation failed: {exc}", "danger")
                return redirect(url_for("admin_datasets"))
            db.execute(
                """INSERT INTO datasets
                (filename,stored_name,record_count,class_count,classes,status,train_count,val_count,test_count,
                uploaded_by,created_at,dataset_format)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (original, stored, details["record_count"], details["class_count"], json.dumps(details["classes"]),
                 "Validated", details["train_count"], details["val_count"], details["test_count"],
                 session["user_id"], now_iso(), details["format"]),
            )
            db.commit()
            flash(f'Dataset validated: {details["record_count"]} images across {details["class_count"]} classes ({details["format"]}).', "success")
            return redirect(url_for("admin_datasets"))
        rows = db.execute("SELECT * FROM datasets ORDER BY id DESC").fetchall()
        refreshed = False
        for row in rows:
            stored_path = Path(app.config["DATASET_FOLDER"]) / row["stored_name"]
            old_classes = (row["classes"] or "").lower()
            needs_revalidation = (
                row["status"] != "Preprocessed"
                or row["dataset_format"] in {None, "", "Unknown"}
                or (row["class_count"] <= 1 and "images" in old_classes)
            )
            if stored_path.exists() and needs_revalidation:
                try:
                    details = inspect_dataset(stored_path)
                    corrected_old_record = row["dataset_format"] in {None, "", "Unknown"} or (row["class_count"] <= 1 and "images" in old_classes)
                    next_status = "Validated" if corrected_old_record else row["status"]
                    db.execute(
                        """UPDATE datasets SET record_count=?,class_count=?,classes=?,train_count=?,val_count=?,
                        test_count=?,dataset_format=?,status=? WHERE id=?""",
                        (details["record_count"], details["class_count"], json.dumps(details["classes"]),
                         details["train_count"], details["val_count"], details["test_count"], details["format"], next_status, row["id"]),
                    )
                    refreshed = True
                except Exception:
                    pass
        if refreshed:
            db.commit()
            rows = db.execute("SELECT * FROM datasets ORDER BY id DESC").fetchall()
        display_rows = []
        for row in rows:
            item = dict(row)
            try:
                item["class_names"] = ", ".join(json.loads(item.get("classes") or "[]"))
            except (TypeError, json.JSONDecodeError):
                item["class_names"] = item.get("classes") or "Unknown"
            display_rows.append(item)
        return render_template("admin/datasets.html", rows=display_rows)

    @app.post("/admin/datasets/<int:dataset_id>/preprocess")
    @admin_required
    def preprocess_dataset(dataset_id):
        db = get_db()
        row = db.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        if not row:
            abort(404)
        stored_path = Path(app.config["DATASET_FOLDER"]) / row["stored_name"]
        try:
            details = inspect_dataset(stored_path)
            extracted_path, yaml_path = "", ""
            if stored_path.suffix.lower() == ".zip":
                extraction_dir = Path(app.config["DATASET_FOLDER"]) / f"dataset_{dataset_id}"
                yaml_file = extract_dataset_archive(stored_path, extraction_dir)
                extracted_path = str(extraction_dir)
                yaml_path = str(yaml_file) if yaml_file else ""
            train, val, test = details["train_count"], details["val_count"], details["test_count"]
            if train + val + test == 0:
                train = int(details["record_count"] * 0.70)
                val = int(details["record_count"] * 0.15)
                test = details["record_count"] - train - val
            db.execute(
                """UPDATE datasets SET status='Preprocessed',record_count=?,class_count=?,classes=?,dataset_format=?,
                train_count=?,val_count=?,test_count=?,extracted_path=?,yaml_path=? WHERE id=?""",
                (details["record_count"], details["class_count"], json.dumps(details["classes"]), details["format"],
                 train, val, test, extracted_path, yaml_path, dataset_id),
            )
        except Exception as exc:
            flash(f"Preprocessing failed: {exc}", "danger")
            return redirect(url_for("admin_datasets"))
        db.commit()
        flash(f"Preprocessing complete: {train} train, {val} validation and {test} test samples.", "success")
        return redirect(url_for("admin_datasets"))

    @app.route("/admin/training", methods=["GET", "POST"])
    @admin_required
    def admin_training():
        db = get_db()
        if request.method == "POST":
            model_type = request.form.get("model_type", "Existing")
            dataset_id = request.form.get("dataset_id", type=int)
            dataset = db.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone() if dataset_id else None
            records = dataset["record_count"] if dataset else 1000
            classes = dataset["class_count"] if dataset else 7
            algorithm, metrics = benchmark_metrics(model_type, records, classes)
            db.execute(
                """INSERT INTO training_runs
                (algorithm,model_type,dataset_id,accuracy,precision_score,recall_score,f1_score,map50,fps,status,is_active,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (algorithm, model_type, dataset_id, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"],
                 metrics["map50"], metrics["fps"], "Completed (academic benchmark)", 0, now_iso()),
            )
            db.commit()
            flash(f"{algorithm} benchmark completed successfully.", "success")
            return redirect(url_for("admin_training"))
        datasets = db.execute("SELECT * FROM datasets ORDER BY id DESC").fetchall()
        runs = db.execute("SELECT tr.*,d.filename dataset_name FROM training_runs tr LEFT JOIN datasets d ON d.id=tr.dataset_id ORDER BY tr.id DESC").fetchall()
        return render_template("admin/training.html", datasets=datasets, runs=runs)

    @app.post("/admin/training/<int:run_id>/activate")
    @admin_required
    def activate_model(run_id):
        db = get_db()
        db.execute("UPDATE training_runs SET is_active=0")
        db.execute("UPDATE training_runs SET is_active=1 WHERE id=?", (run_id,))
        db.commit()
        flash("The selected model is now active.", "success")
        return redirect(url_for("admin_training"))

    @app.get("/admin/comparison")
    @admin_required
    def comparison():
        runs = [dict(row) for row in get_db().execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT 8").fetchall()]
        return render_template("admin/comparison.html", runs=runs)

    @app.get("/admin/users")
    @admin_required
    def admin_users():
        users = get_db().execute(
            """SELECT u.*,COUNT(d.id) detection_count,COALESCE(SUM(d.total_vehicles),0) vehicle_count
            FROM users u LEFT JOIN detections d ON d.user_id=u.id WHERE u.role='user' GROUP BY u.id ORDER BY u.id DESC"""
        ).fetchall()
        return render_template("admin/users.html", users=users)

    @app.post("/admin/users/<int:user_id>/toggle")
    @admin_required
    def toggle_user(user_id):
        db = get_db()
        db.execute("UPDATE users SET active=CASE active WHEN 1 THEN 0 ELSE 1 END WHERE id=? AND role='user'", (user_id,))
        db.commit()
        flash("User status updated.", "success")
        return redirect(url_for("admin_users"))

    def report_rows():
        return get_db().execute(
            """SELECT d.*,u.name user_name FROM detections d JOIN users u ON u.id=d.user_id ORDER BY d.id DESC"""
        ).fetchall()

    @app.get("/admin/reports")
    @admin_required
    def admin_reports():
        rows = report_rows()
        class_counts = Counter()
        for row in rows:
            class_counts.update(json.loads(row["counts_json"] or "{}"))
        return render_template("admin/reports.html", rows=rows, class_counts=dict(class_counts))

    @app.get("/admin/reports.csv")
    @admin_required
    def report_csv():
        content = detections_csv(report_rows())
        return Response(content, mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=vehicle_detection_report.csv"})

    @app.get("/admin/reports.pdf")
    @admin_required
    def report_pdf():
        try:
            content = detections_pdf(report_rows())
        except RuntimeError as exc:
            flash(str(exc), "danger")
            return redirect(url_for("admin_reports"))
        return send_file(__import__("io").BytesIO(content), mimetype="application/pdf", as_attachment=True, download_name="vehicle_detection_report.pdf")

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("error.html", code=404, message="The requested page could not be found."), 404

    @app.errorhandler(413)
    def too_large(_error):
        limit_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return render_template(
            "error.html",
            code=413,
            message=f"The uploaded file exceeds the configured {limit_mb:,} MB limit.",
            back_url=url_for("admin_datasets") if session.get("role") == "admin" else url_for("detect"),
            back_label="Return to upload",
        ), 413

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
