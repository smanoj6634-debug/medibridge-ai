def _serve_asset(kind, ver):
    a = ASSETS[kind]
    if ver != a["ver"]:
        abort(404)
    etag = '"%s"' % a["ver"]
    if request.headers.get("If-None-Match") == etag:
        resp = app.response_class(status=304)
    else:
        gz = "gzip" in request.headers.get("Accept-Encoding", "").lower()
        resp = app.response_class(a["gz"] if gz else a["raw"], mimetype=a["mimetype"])
        if gz:
            resp.headers["Content-Encoding"] = "gzip"
    resp.headers.update({"ETag": etag, "Cache-Control": "public, max-age=31536000, immutable", "Vary": "Accept-Encoding"})
    return resp


@app.route("/assets/app.<ver>.css")
def asset_css(ver):
    return _serve_asset("css", ver)


@app.route("/assets/app.<ver>.js")
def asset_js(ver):
    return _serve_asset("js", ver)


@app.context_processor
def inject_globals():
    return {"disclaimer": DISCLAIMER, "ai_ready": ai_configured(), "asset_css": ASSET_URLS["css"], "asset_js": ASSET_URLS["js"]}


# ---------- Public pages ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if g.user:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        pw = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        agree = request.form.get("agree")
        error = None
        if len(name) < 2:
            error = "Enter your full name."
        elif not EMAIL_RE.match(email):
            error = "Enter a valid email address."
        elif len(pw) < 8:
            error = "Password must be at least 8 characters."
        elif pw != confirm:
            error = "Passwords do not match."
        elif not agree:
            error = "Please accept the disclaimer and terms to continue."
        if not error and not ip_rate_ok("signup", 10, 3600):
            error = "Too many sign-ups from this network. Please try again later."
        if not error:
            try:
                query("INSERT INTO users (name, email, password_hash) VALUES (?,?,?)",
                      (name, email, generate_password_hash(pw)), commit=True)
                new = query("SELECT id FROM users WHERE email = ?", (email,), one=True)
                audit("signup", user_id=new["id"] if new else None)
                flash("Account created. Please log in.", "success")
                return redirect(url_for("login"))
            except IntegrityError:
                get_db().rollback()
                error = "An account with this email already exists."
        flash(error, "error")
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        pw = request.form.get("password", "")
        if not ip_rate_ok("login", 10, 900):
            flash("Too many log-in attempts. Please wait 15 minutes and try again.", "error")
            return render_template("login.html"), 429
        user = query("SELECT * FROM users WHERE email = ?", (email,), one=True)
        if user and check_password_hash(user["password_hash"], pw):
            session.clear()
            session["user_id"] = user["id"]
            audit("login", user_id=user["id"])
            return redirect(safe_next(request.args.get("next")) or url_for("dashboard"))
        if user:
            audit("login_failed", user_id=user["id"])
        flash("Incorrect email or password.", "error")
    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    if g.user:
        audit("logout")
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("index"))


# ---------- App pages (login required) ----------
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", features=feature_list(), live=live_status(), labels=STATUS_LABEL,
                           consents=get_consents(), purposes=CONSENT_PURPOSES,
                           device_types=DEVICE_TYPES, reg_status=REG_STATUS,
                           reading_metrics=READING_METRICS)


@app.route("/twin")
@login_required
def twin():
    return render_template("twin.html", patient_id="MB-PT-%06d" % g.user["id"])


@app.route("/report")
@login_required
def report():
    return render_template("report.html")


@app.route("/reminders")
@login_required
def reminders():
    return render_template("reminders.html")


@app.route("/assistant")
@login_required
def assistant():
    return render_template("assistant.html")


# ---------- Simple per-user rate limit (protects your API bill) ----------
import time
_RATE = {}
_IP_RATE = _RATE                                  # old name kept: one shared table for user and address limits


def _hit(bucket, key, limit, window):
    """Sliding-window limiter. True if this call is allowed (and counted)."""
    now = time.time()
    k = (bucket, str(key))
    hits = [t for t in _RATE.get(k, ()) if now - t < window]
    allowed = len(hits) < limit
    if allowed:
        hits.append(now)
    _RATE[k] = hits
    if len(_RATE) > 4000:                         # forget idle keys only. Never clear active ones, or an attacker could reset every limit
        for stale in [s for s, v in _RATE.items() if not v or now - v[-1] > 3600]:
            del _RATE[stale]
    return allowed


def rate_ok(bucket, limit, window=3600):
    return _hit(bucket, g.user["id"], limit, window)


# ---------- JSON APIs ----------
@app.route("/api/report", methods=["POST"])
@login_required
def api_report():
    f = request.files.get("report")
    if not f or not f.filename:
        return jsonify(error="Choose a file to upload."), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify(error="Unsupported file type. Use PDF, JPG, PNG or TXT."), 400
    if request.form.get("ack") != "1":
        return jsonify(error="Please confirm you removed personal details from the report."), 400
    if ai_configured():
        blocked = consent_blocked("ai_processing", "AI processing")
        if blocked:
            return blocked
    if ai_configured() and not rate_ok("report", 10):
        return jsonify(error="Report limit reached (10 per hour). Please try again later."), 429
    lang = request.form.get("lang", "English")
    lang = lang if lang in ("English", "Tamil", "Hindi") else "English"
    data = f.read()  # processed in memory; reports are NOT saved to disk
    try:
        result = explain_report(os.path.basename(f.filename), data, lang)
        audit("ai_report")
        return jsonify(result)
    except AIError as e:
        app.logger.warning("AI report error: %s", str(e)[:200])
        return jsonify(error=str(e)), 502


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    body = request.get_json(silent=True) or {}
    msg = str(body.get("message", "")).strip()
    if not msg:
        return jsonify(error="Type a question first."), 400
    if len(msg) > 1000:
        return jsonify(error="Please keep your question under 1000 characters."), 400
    mode = body.get("mode", "chat")
    mode = mode if mode in ("chat", "term", "appointment") else "chat"
    lang = body.get("lang", "English")
    lang = lang if lang in LANGS else "English"
    if ai_configured() and not is_emergency(msg):
        blocked = consent_blocked("ai_processing", "AI processing")
        if blocked:
            return blocked
    if ai_configured() and not rate_ok("chat", 40):
        return jsonify(error="Message limit reached (40 per hour). Please try again later."), 429
    try:
        result = chat_reply(msg, body.get("history", []), mode, lang)
        if not result.get("emergency"):
            audit("ai_chat", mode)
        return jsonify(result)
    except AIError as e:
        app.logger.warning("AI chat error: %s", str(e)[:200])
        return jsonify(error=str(e)), 502


# ---------- Platform status, consent, audit and data rights ----------
STATUS_LABEL = {"impl": "Implemented", "part": "Partial", "demo": "Demo data", "exp": "Experimental", "plan": "Planned"}

CONSENT_PURPOSES = {
    "ai_processing": {"title": "AI processing",
                      "desc": "Send my questions and uploaded reports to the AI provider so it can write explanations. Live Medical Research also sends the topic I type to a web-search provider (Olostep). MediBridge AI does not save my questions."},
    "push_alerts": {"title": "Reminder alerts",
                    "desc": "Store a push address for my browser or phone so reminder alerts can be sent to it."},
}


def feature_list():
    ai_note = "AI key connected." if ai_configured() else "Needs GEMINI_API_KEY or AI_API_KEY on the server."
    return [
        dict(name="Account, login and sessions", status="impl", note="Hashed passwords, CSRF tokens, secure cookies, log-in rate limit."),
        dict(name="AI Health Companion", status="impl" if ai_configured() else "part", note="Education only, never diagnoses. " + ai_note),
        dict(name="Smart appointment preparation", status="impl" if ai_configured() else "part", note="AI drafts questions to ask. " + ai_note),
        dict(name="Live Medical Research", status="impl" if research_configured() else "part",
             note="Olostep searches public sources, preferring WHO, CDC, NHS and government sites. The AI explains them with links and a last-checked date. Results are cached to save credits. "
                  + ("Connected." if research_configured() else "Needs OLOSTEP_API_KEY and the AI key on the server.")),
        dict(name="Medical Report Explainer", status="part", note="Uploads are explained by the AI provider and are not clinically validated. The sample is fictional."),
        dict(name="Medicine reminders and push alerts", status="part", note="Server scheduler is built. Delivery depends on the phone and must be confirmed on your own device."),
        dict(name="Patient consent management", status="impl", note="Stored per account and enforced for AI processing and alerts."),
        dict(name="Audit log, data export and deletion", status="impl", note="Activity kept 180 days. You can download or delete everything."),
        dict(name="Emergency information", status="impl", note="Your contact is kept only in this browser."),
        dict(name="Digital Health Timeline", status="part", note="Reminders and entries with a phone alert are real and saved to your account. Other entries are demo samples or browser-only notes."),
        dict(name="Timeline phone alerts", status="part", note="Server scheduler sends a push at the time you choose. Needs alerts turned on and the server awake. Delivery must be confirmed on your own device."),
        dict(name="Health Knowledge Hub", status="part", note="Topic shortcuts open AI explanations. Needs the AI key."),
        dict(name="Explainable AI insights", status="part", note="Range checks show the exact rule used. No predictive model exists yet."),
        dict(name="Smart Medical Device Hub", status="part", note="Supports Web Bluetooth LE pairing, ESP32 HTTP API ingestion, and clearly labelled demo mode."),
        dict(name="Medical Device Digital Passport", status="impl", note="Record model, firmware and calibration dates per device. Self-reported and not verified."),
        dict(name="Manual health readings and trend chart", status="impl", note="Type in readings, see charts, statistics and a fitted line. Describes your entries only. Does not diagnose or predict."),
        dict(name="Trend analysis with AI", status="part", note="Statistical trends are live. The AI only helps you prepare doctor questions. AI-written interpretation is not built and needs clinical validation."),
        dict(name="Sample charts (demo)", status="demo", note="The sample charts use simulated data and say so on the chart."),
        dict(name="Tamil voice assistant", status="exp", note="Uses the browser's read-aloud and dictation. Quality depends on the phone's Tamil voices."),
        dict(name="Bluetooth device integration", status="part", note="Web Bluetooth GATT integration built for Heart Rate, SpO2, Thermometer, BP, and ESP32 BLE JSON in Chrome/Edge."),
        dict(name="Digital Patient Twin", status="part", note="Interactive 3D Three.js body with organ views, BLE & ESP32 live stream, trend graphs, and Gemini AI summary. Educational prototype; not clinically validated."),
        dict(name="Secure healthcare data sharing", status="plan", note="Time-limited, consented sharing with a named clinician. Needs access controls and a security review first."),
    ]


def live_status():
    prov = ai_provider()
    return [
        ("AI service", bool(prov), ("connected (%s)" % prov) if prov else "not connected"),
        ("Research search", olostep_configured(), "connected (Olostep)" if olostep_configured() else "not connected"),
        ("Database", True, "PostgreSQL" if USE_PG else "SQLite (local)"),
        ("Push library", PUSH_OK, "ready" if PUSH_OK else "pywebpush missing"),
        ("Reminder scheduler", _SCHED["started"], "running" if _SCHED["started"] else "off"),
    ]


def get_consents():
    return {r["purpose"]: bool(r["granted"]) for r in
            query_all("SELECT purpose, granted FROM consents WHERE user_id = ?", (g.user["id"],))}


def has_consent(purpose):
    row = query("SELECT granted FROM consents WHERE user_id = ? AND purpose = ?", (g.user["id"], purpose), one=True)
    return bool(row and row["granted"])


def consent_blocked(purpose, what):
    """JSON error telling the page to ask for consent. Returns None when consent exists."""
    if has_consent(purpose):
        return None
    return jsonify(error="Please allow %s in the Privacy centre first." % what, code="consent_required"), 403


def _utcnow():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def audit(event, detail="", user_id=None):
    """Record a security-relevant event. Never put message or report content in here."""
    try:
        uid = user_id if user_id is not None else (g.user["id"] if g.get("user") else None)
        query("INSERT INTO audit_log (user_id, event, detail, created_at) VALUES (?, ?, ?, ?)",
              (uid, event[:40], str(detail)[:80], _utcnow()), commit=True)
    except Exception as e:
        app.logger.warning("audit failed: %s", type(e).__name__)
        try:
            get_db().rollback()
        except Exception:
            pass


def purge_audit():
    """Retention policy: audit records older than 180 days are deleted."""
    try:
        with app.app_context():
            cutoff = (datetime.utcnow() - timedelta(days=180)).strftime("%Y-%m-%d %H:%M:%S")
            query("DELETE FROM audit_log WHERE created_at < ?", (cutoff,), commit=True)
    except Exception as e:
        print("Audit purge skipped:", type(e).__name__)


def ip_rate_ok(bucket, limit, window):
    """Rate limit by network address (for log-in and sign-up, where nobody is logged in yet)."""
    return _hit(bucket, request.remote_addr or "?", limit, window)


@app.route("/api/consents", methods=["GET"])
@login_required
def api_consents_list():
    return jsonify(consents=get_consents())


@app.route("/api/consents", methods=["POST"])
@login_required
def api_consents_set():
    body = request.get_json(silent=True) or {}
    purpose, granted = body.get("purpose"), bool(body.get("granted"))
    if purpose not in CONSENT_PURPOSES:
        return jsonify(error="Unknown consent type."), 400
    if not rate_ok("consent", 60):
        return jsonify(error="Too many changes. Please try again later."), 429
    uid = g.user["id"]
    query("DELETE FROM consents WHERE user_id = ? AND purpose = ?", (uid, purpose))
    query("INSERT INTO consents (user_id, purpose, granted, updated_at) VALUES (?, ?, ?, ?)",
          (uid, purpose, 1 if granted else 0, _utcnow()), commit=True)
    if purpose == "push_alerts" and not granted:          # withdrawing consent removes the stored push addresses
        query("DELETE FROM push_subs WHERE user_id = ?", (uid,), commit=True)
    audit("consent_granted" if granted else "consent_withdrawn", purpose)
    return jsonify(consents=get_consents())


@app.route("/api/audit")
@login_required
def api_audit():
    rows = query_all("SELECT event, detail, created_at FROM audit_log WHERE user_id = ? ORDER BY id DESC LIMIT 20",
                     (g.user["id"],))
    return jsonify(events=[{"event": r["event"], "detail": r["detail"], "created_at": str(r["created_at"])} for r in rows])


@app.route("/api/privacy/export")
@login_required
def api_privacy_export():
    uid = g.user["id"]
    acct = query("SELECT name, email, created_at FROM users WHERE id = ?", (uid,), one=True)
    data = {
        "exported_at_utc": _utcnow(),
        "account": {"name": acct["name"], "email": acct["email"], "created_at": str(acct["created_at"])},
        "reminders": [dict(r, created_at=str(r["created_at"])) for r in query_all(
            "SELECT name, remind_time, note, created_at FROM reminders WHERE user_id = ? ORDER BY id", (uid,))],
        "consents": [dict(r) for r in query_all("SELECT purpose, granted, updated_at FROM consents WHERE user_id = ?", (uid,))],
        "device_passports": [dict(r) for r in query_all(
            "SELECT label, dtype, maker, model, serial_tail, firmware, first_use, last_cal, next_cal, reg_status, notes "
            "FROM device_passports WHERE user_id = ? ORDER BY id", (uid,))],
        "twin_esp32_tokens": [dict(r) for r in query_all(
            "SELECT label, token_prefix, last_seen, created_at FROM twin_tokens WHERE user_id = ? ORDER BY id", (uid,))],
        "health_readings": [dict(r) for r in query_all(
            "SELECT metric, value, value2, measured_at AS measured_at_utc, context, note, source FROM health_readings "
            "WHERE user_id = ? ORDER BY measured_at", (uid,))],
        "reading_ranges": [dict(r) for r in query_all("SELECT metric, low, high FROM reading_ranges WHERE user_id = ?", (uid,))],
        "timeline_alerts": [dict(r) for r in query_all(
            "SELECT etype AS type, title, when_utc, notify_min FROM timeline_events WHERE user_id = ? ORDER BY when_utc", (uid,))],
        "alert_devices_registered": len(query_all("SELECT id FROM push_subs WHERE user_id = ?", (uid,))),
        "activity_log": [dict(r, created_at=str(r["created_at"])) for r in query_all(
            "SELECT event, detail, created_at FROM audit_log WHERE user_id = ? ORDER BY id", (uid,))],
        "not_stored": "Chat messages and uploaded reports are not stored by MediBridge AI.",
    }
    audit("data_export")
    resp = jsonify(data)
    resp.headers["Content-Disposition"] = "attachment; filename=medibridge-my-data.json"
    return resp


@app.route("/api/privacy/delete", methods=["POST"])
@login_required
def api_privacy_delete():
    body = request.get_json(silent=True) or {}
    if not rate_ok("delete", 5):
        return jsonify(error="Too many attempts. Please try again later."), 429
    row = query("SELECT password_hash FROM users WHERE id = ?", (g.user["id"],), one=True)
    if not row or not check_password_hash(row["password_hash"], str(body.get("password", ""))):
        audit("delete_failed", "wrong password")
        return jsonify(error="Incorrect password."), 403
    uid = g.user["id"]
    for sql in ("DELETE FROM reminder_log WHERE reminder_id IN (SELECT id FROM reminders WHERE user_id = ?)",
                "DELETE FROM reminders WHERE user_id = ?", "DELETE FROM push_subs WHERE user_id = ?",
                "DELETE FROM consents WHERE user_id = ?", "DELETE FROM device_passports WHERE user_id = ?",
                "DELETE FROM twin_tokens WHERE user_id = ?",
                "DELETE FROM event_alerts_sent WHERE event_id IN (SELECT id FROM timeline_events WHERE user_id = ?)",
                "DELETE FROM timeline_events WHERE user_id = ?", "DELETE FROM health_readings WHERE user_id = ?", "DELETE FROM reading_ranges WHERE user_id = ?",
                "DELETE FROM audit_log WHERE user_id = ?"):
        query(sql, (uid,))
    query("DELETE FROM users WHERE id = ?", (uid,), commit=True)      # one transaction for everything
    session.clear()
    app.logger.info("account %s deleted by owner", uid)
    return jsonify(ok=True)


# ---------- Medical Device Digital Passport (self-reported device records) ----------
DEVICE_TYPES = {"pulse_oximeter": "Pulse oximeter", "heart_rate": "Heart-rate sensor", "thermometer": "Thermometer",
                "bp_monitor": "Blood pressure monitor", "glucose_meter": "Glucose meter",
                "respiratory": "Respiratory sensor", "other": "Other device"}
REG_STATUS = {"unknown": "Not known", "approved": "Approved or cleared in my country", "ce": "CE marked",
              "fda": "FDA cleared", "wellness": "Wellness device, not medical"}
MAX_DEVICES = 10


def _clean_date(v):
    """'' for empty, 'YYYY-MM-DD' for a valid date, None for anything else."""
    v = str(v or "").strip()
    if not v:
        return ""
    try:
        return datetime.strptime(v, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


def _cal_status(next_cal):
    if not next_cal:
        return "not_set"
    days = (datetime.strptime(next_cal, "%Y-%m-%d").date() - datetime.utcnow().date()).days
    return "overdue" if days < 0 else ("due_soon" if days <= 30 else "ok")


def _user_devices():
    rows = query_all(
        "SELECT id, label, dtype, maker, model, serial_tail, firmware, first_use, last_cal, next_cal, reg_status, notes "
        "FROM device_passports WHERE user_id = ? ORDER BY id DESC", (g.user["id"],))
    for r in rows:
        r["cal_status"] = _cal_status(r["next_cal"])
    return rows


@app.route("/api/devices", methods=["GET"])
@login_required
def api_devices_list():
    return jsonify(devices=_user_devices())


@app.route("/api/devices", methods=["POST"])
@login_required
def api_devices_add():
    b = request.get_json(silent=True) or {}
    label = str(b.get("label", "")).strip()[:60]
    if not label:
        return jsonify(error="Give the device a name."), 400
    dtype, reg = b.get("dtype", "other"), b.get("reg_status", "unknown")
    if dtype not in DEVICE_TYPES or reg not in REG_STATUS:
        return jsonify(error="Choose a valid device type and approval status."), 400
    dates = [_clean_date(b.get(k)) for k in ("first_use", "last_cal", "next_cal")]
    if None in dates:
        return jsonify(error="Dates must look like 2026-11-21."), 400
    if not rate_ok("device", 40):
        return jsonify(error="Too many changes. Please try again later."), 429
    if len(_user_devices()) >= MAX_DEVICES:
        return jsonify(error="You can save up to %d device passports. Delete one first." % MAX_DEVICES), 400
    serial_tail = re.sub(r"[^A-Za-z0-9]", "", str(b.get("serial_tail", "")))[-4:]      # last 4 only (data minimisation)
    query("INSERT INTO device_passports (user_id, label, dtype, maker, model, serial_tail, firmware, first_use, "
          "last_cal, next_cal, reg_status, notes, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
          (g.user["id"], label, dtype, str(b.get("maker", "")).strip()[:60], str(b.get("model", "")).strip()[:60],
           serial_tail, str(b.get("firmware", "")).strip()[:30], dates[0], dates[1], dates[2], reg,
           str(b.get("notes", "")).strip()[:200], _utcnow()), commit=True)
    audit("device_added", dtype)
    return jsonify(devices=_user_devices())


@app.route("/api/devices/<int:did>/delete", methods=["POST"])
@login_required
def api_devices_delete(did):
    query("UPDATE health_readings SET device_id = NULL WHERE device_id = ? AND user_id = ?", (did, g.user["id"]))
    query("DELETE FROM device_passports WHERE id = ? AND user_id = ?", (did, g.user["id"]), commit=True)
    audit("device_deleted")
    return jsonify(devices=_user_devices())


# ---------- Manual health readings and statistical trends ----------
# These routes only store and describe numbers the user typed in. They never label a value as good or bad,
# diagnose, or predict. "Plausible range" checks below only catch typing mistakes.
READING_METRICS = {
    "bp": {"label": "Blood pressure", "unit": "mmHg", "two": True, "lo": 50, "hi": 260, "lo2": 30, "hi2": 160},
    "heart_rate": {"label": "Heart rate", "unit": "bpm", "two": False, "lo": 20, "hi": 250},
    "spo2": {"label": "Oxygen saturation (SpO2)", "unit": "%", "two": False, "lo": 50, "hi": 100},
    "temperature": {"label": "Body temperature", "unit": "°C", "two": False, "lo": 30, "hi": 45},
    "glucose": {"label": "Blood glucose", "unit": "mg/dL", "two": False, "lo": 20, "hi": 800},
    "weight": {"label": "Weight", "unit": "kg", "two": False, "lo": 2, "hi": 400},
    "resp_rate": {"label": "Respiratory rate", "unit": "breaths/min", "two": False, "lo": 4, "hi": 60},
}
RANGE_KEYS = {"bp_sys", "bp_dia", "heart_rate", "spo2", "temperature", "glucose", "weight", "resp_rate"}
READING_CONTEXTS = {"", "fasting", "after_meal", "resting", "after_activity"}
MAX_READINGS = 2000
# two-sided 95% critical t values by degrees of freedom (largest listed df not above the real df is used)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}
_TS = "%Y-%m-%d %H:%M:%S"


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _parse_when(s):
    """ISO time from the browser -> naive UTC datetime, or None if invalid / unreasonable."""
    try:
        d = datetime.fromisoformat(str(s).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if d.tzinfo is not None:
        d = (d - d.utcoffset()).replace(tzinfo=None)
    now = datetime.utcnow()
    if d > now + timedelta(minutes=10) or d < now - timedelta(days=3650):
        return None
    return d.replace(microsecond=0)


def _iso(ts):
    return datetime.strptime(ts, _TS).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_ranges():
    return {r["metric"]: {"low": r["low"], "high": r["high"]}
            for r in query_all("SELECT metric, low, high FROM reading_ranges WHERE user_id = ?", (g.user["id"],))}


def _trend_series(rows, key, label, unit, field):
    """Plain descriptive statistics plus a least-squares line. No prediction."""
    pts = [(datetime.strptime(r["measured_at"], _TS), float(r[field])) for r in rows if r[field] is not None]
    out = {"key": key, "label": label, "unit": unit, "stats": None, "fit": None, "note": ""}
    if not pts:
        out["note"] = "No readings in this period."
        return out
    vals, n, t0 = [v for _, v in pts], len(pts), pts[0][0]
    xs = [(t - t0).total_seconds() / 86400 for t, _ in pts]
    mean = sum(vals) / n
    st = {"n": n, "latest": vals[-1], "latest_at": pts[-1][0].strftime("%Y-%m-%dT%H:%M:%SZ"),
          "min": min(vals), "max": max(vals), "mean": round(mean, 1)}
    span = xs[-1]
    if n >= 5 and span >= 1:
        mx = sum(xs) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (v - mean) for x, v in zip(xs, vals)) / sxx
        intercept = mean - slope * mx
        change = slope * span
        sse = sum((v - (intercept + slope * x)) ** 2 for x, v in zip(xs, vals))
        if sse < 1e-9:                                  # points lie exactly on a straight line
            direction = "no clear direction" if abs(slope) < 1e-9 else ("rising" if slope > 0 else "falling")
        else:                                           # a direction is named only if the slope clearly beats the scatter (t-test, 95%)
            se = math.sqrt(sse / (n - 2) / sxx)
            crit = 1.96
            for k in sorted(_T95):
                if k <= n - 2:
                    crit = _T95[k]
            direction = "no clear direction" if abs(slope / se) < crit else ("rising" if slope > 0 else "falling")
        st.update(direction=direction, slope_per_week=round(slope * 7, 2), fitted_change=round(change, 1))
        out["fit"] = {"t0": pts[0][0].strftime("%Y-%m-%dT%H:%M:%SZ"), "v0": round(intercept, 2),
                      "t1": pts[-1][0].strftime("%Y-%m-%dT%H:%M:%SZ"), "v1": round(intercept + slope * span, 2)}
    else:
        out["note"] = "Add at least 5 readings over 2 or more days to see a trend line."
    out["stats"] = st
    return out


def _reading_rows(metric="", days=None, limit=MAX_READINGS):
    sql = ("SELECT r.id, r.metric, r.value, r.value2, r.measured_at, r.context, r.note, r.device_id, d.label AS device_label "
           "FROM health_readings r LEFT JOIN device_passports d ON d.id = r.device_id AND d.user_id = r.user_id "
           "WHERE r.user_id = ?")
    params = [g.user["id"]]
    if metric:
        sql += " AND r.metric = ?"
        params.append(metric)
    if days:
        sql += " AND r.measured_at >= ?"
        params.append((datetime.utcnow() - timedelta(days=days)).strftime(_TS))
    sql += " ORDER BY r.measured_at ASC, r.id ASC LIMIT %d" % int(limit)
    return query_all(sql, tuple(params))


@app.route("/api/readings", methods=["GET"])
@login_required
def api_readings_list():
    metric = request.args.get("metric", "")
    if metric and metric not in READING_METRICS:
        return jsonify(error="Unknown measurement."), 400
    raw = request.args.get("days", "30")
    days = None
    if raw != "all":
        try:
            days = int(raw)
        except ValueError:
            return jsonify(error="Invalid period."), 400
        if not 1 <= days <= 3650:
            return jsonify(error="Invalid period."), 400
    rows = _reading_rows(metric, days)
    readings = [dict(r, measured_at=_iso(r["measured_at"])) for r in rows]
    resp = {"readings": readings, "series": [], "ranges": _get_ranges()}
    if metric:
        m = READING_METRICS[metric]
        resp.update(metric=metric, label=m["label"], unit=m["unit"])
        if m["two"]:
            resp["series"] = [_trend_series(rows, "bp_sys", "Systolic", m["unit"], "value"),
                              _trend_series(rows, "bp_dia", "Diastolic", m["unit"], "value2")]
        else:
            resp["series"] = [_trend_series(rows, metric, m["label"], m["unit"], "value")]
    return jsonify(resp)


@app.route("/api/readings", methods=["POST"])
@login_required
def api_readings_add():
    b = request.get_json(silent=True) or {}
    metric = b.get("metric")
    if metric not in READING_METRICS:
        return jsonify(error="Choose a measurement type."), 400
    m = READING_METRICS[metric]
    v = _num(b.get("value"))
    if v is None:
        return jsonify(error="Enter the reading as a number."), 400
    unit = str(b.get("unit", ""))
    if metric == "temperature" and unit == "F":
        v = (v - 32) * 5 / 9
    elif metric == "glucose" and unit == "mmol/L":
        v = v * 18.016
    v = round(v, 1)
    v2 = None
    if m["two"]:
        v2 = _num(b.get("value2"))
        if v2 is None:
            return jsonify(error="Enter both blood pressure numbers."), 400
        v2 = round(v2, 1)
        if not m["lo2"] <= v2 <= m["hi2"] or v <= v2:
            return jsonify(error="Please check the blood pressure numbers. The top number should be larger than the bottom one."), 400
    if not m["lo"] <= v <= m["hi"]:
        return jsonify(error="That value looks like a typing mistake for %s. Please check it." % m["label"].lower()), 400
    when = _parse_when(b.get("measured_at"))
    if when is None:
        return jsonify(error="Choose a valid date and time (not in the future)."), 400
    ctx = str(b.get("context", "") or "")
    if ctx not in READING_CONTEXTS:
        return jsonify(error="Unknown situation."), 400
    dev = b.get("device_id")
    dev_id = None
    if dev not in (None, ""):
        try:
            dev_id = int(dev)
        except (TypeError, ValueError):
            return jsonify(error="Unknown device."), 400
        if not query("SELECT 1 FROM device_passports WHERE id = ? AND user_id = ?", (dev_id, g.user["id"]), one=True):
            return jsonify(error="Unknown device."), 400
    if not rate_ok("reading", 120):
        return jsonify(error="Too many readings this hour. Please try again later."), 429
    if query("SELECT COUNT(*) AS n FROM health_readings WHERE user_id = ?", (g.user["id"],), one=True)["n"] >= MAX_READINGS:
        return jsonify(error="You reached the limit of %d readings. Download your data and delete old readings." % MAX_READINGS), 400
    query("INSERT INTO health_readings (user_id, metric, value, value2, measured_at, context, note, device_id, source, created_at) "
          "VALUES (?,?,?,?,?,?,?,?,?,?)",
          (g.user["id"], metric, v, v2, when.strftime(_TS), ctx, str(b.get("note", "")).strip()[:120], dev_id, "manual", _utcnow()),
          commit=True)
    _twin_notify(g.user["id"])
    audit("reading_added", metric)
    return jsonify(ok=True)


@app.route("/api/readings/<int:rid>/delete", methods=["POST"])
@login_required
def api_readings_delete(rid):
    query("DELETE FROM health_readings WHERE id = ? AND user_id = ?", (rid, g.user["id"]), commit=True)
    audit("reading_deleted")
    return jsonify(ok=True)


@app.route("/api/ranges", methods=["GET"])
@login_required
def api_ranges_get():
    return jsonify(ranges=_get_ranges())


@app.route("/api/ranges", methods=["POST"])
@login_required
def api_ranges_set():
    """A reference range the user copied from their own doctor or report. Used only to draw a band and compare."""
    b = request.get_json(silent=True) or {}
    key = b.get("metric")
    if key not in RANGE_KEYS:
        return jsonify(error="Unknown measurement."), 400
    low = _num(b.get("low")) if str(b.get("low", "")).strip() != "" else None
    high = _num(b.get("high")) if str(b.get("high", "")).strip() != "" else None
    if (str(b.get("low", "")).strip() != "" and low is None) or (str(b.get("high", "")).strip() != "" and high is None):
        return jsonify(error="Enter the range as numbers."), 400
    if low is not None and high is not None and low > high:
        return jsonify(error="The lower number must not be larger than the upper number."), 400
    if not rate_ok("range", 60):
        return jsonify(error="Too many changes. Please try again later."), 429
    query("DELETE FROM reading_ranges WHERE user_id = ? AND metric = ?", (g.user["id"], key))
    if low is None and high is None:
        get_db().commit()
    else:
        query("INSERT INTO reading_ranges (user_id, metric, low, high) VALUES (?,?,?,?)", (g.user["id"], key, low, high), commit=True)
    audit("range_saved", key)
    return jsonify(ranges=_get_ranges())


def _csv_safe(s):
    s = str(s or "")
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


@app.route("/api/readings/export.csv")
@login_required
def api_readings_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["measured_at_utc", "measurement", "value", "value_2", "unit", "situation", "device", "note", "source"])
    for r in _reading_rows():
        m = READING_METRICS.get(r["metric"], {"label": r["metric"], "unit": ""})
        w.writerow([r["measured_at"], m["label"], r["value"], "" if r["value2"] is None else r["value2"], m["unit"],
                    _csv_safe(r["context"]), _csv_safe(r["device_label"]), _csv_safe(r["note"]), "manual entry"])
    audit("readings_export")
    return app.response_class(buf.getvalue(), mimetype="text/csv",
                              headers={"Content-Disposition": "attachment; filename=medibridge-readings.csv"})


# ---------- Digital Patient Twin: Real-Time SSE, BLE Save, ESP32 Ingestion & AI Analysis ----------
TWIN_METRICS = ("heart_rate", "spo2", "temperature", "resp_rate", "bp")
_TWIN_SUBS = {}
_TWIN_LOCK = threading.Lock()


def _twin_notify(uid):
    with _TWIN_LOCK:
        queues = list(_TWIN_SUBS.get(uid, ()))
    for q in queues:
        try:
            q.put_nowait({"ts": _utcnow()})
        except queue.Full:
            pass


def _save_sensor_reading(uid, metric, val, val2=None, measured_at=None, source="ble", note=""):
    if metric not in TWIN_METRICS:
        return False, "Unsupported Twin metric."
    m = READING_METRICS[metric]
    v = _num(val)
    if v is None:
        return False, "Value must be numeric."
    v = round(v, 1)
    v2 = None
    if m["two"]:
        v2 = _num(val2)
        if v2 is None:
            return False, "Both systolic and diastolic values are required."
        v2 = round(v2, 1)
        if not m["lo2"] <= v2 <= m["hi2"] or v <= v2:
            return False, "Invalid blood pressure numbers."
    if not m["lo"] <= v <= m["hi"]:
        return False, "Value out of plausible range for %s." % m["label"].lower()
    when = _parse_when(measured_at) if measured_at else datetime.utcnow().replace(microsecond=0)
    if when is None:
        when = datetime.utcnow().replace(microsecond=0)
    if query("SELECT COUNT(*) AS n FROM health_readings WHERE user_id = ?", (uid,), one=True)["n"] >= MAX_READINGS:
        oldest = query("SELECT id FROM health_readings WHERE user_id = ? ORDER BY measured_at ASC, id ASC LIMIT 1", (uid,), one=True)
        if oldest:
            query("DELETE FROM health_readings WHERE id = ?", (oldest["id"],))
    query("INSERT INTO health_readings (user_id, metric, value, value2, measured_at, context, note, device_id, source, created_at) "
          "VALUES (?,?,?,?,?,?,?,?,?,?)",
          (uid, metric, v, v2, when.strftime(_TS), "", str(note)[:120], None, source[:16], _utcnow()), commit=True)
    return True, None


def _twin_snapshot(uid, days=1):
    latest = {}
    history = {}
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime(_TS)
    for k in TWIN_METRICS:
        last_row = query(
            "SELECT value, value2, measured_at, source FROM health_readings "
            "WHERE user_id = ? AND metric = ? ORDER BY measured_at DESC, id DESC LIMIT 1",
            (uid, k), one=True)
        if last_row:
            latest[k] = {"value": last_row["value"], "value2": last_row["value2"],
                         "measured_at": _iso(last_row["measured_at"]), "source": last_row["source"] or "manual"}
        rows = query_all(
            "SELECT value, value2, measured_at, source FROM health_readings "
            "WHERE user_id = ? AND metric = ? AND measured_at >= ? ORDER BY measured_at ASC, id ASC LIMIT 200",
            (uid, k, cutoff))
        history[k] = [{"value": r["value"], "value2": r["value2"], "measured_at": _iso(r["measured_at"]),
                       "source": r["source"] or "manual"} for r in rows]
    recent_rows = query_all(
        "SELECT metric, value, value2, measured_at, source FROM health_readings "
        "WHERE user_id = ? AND metric IN ('heart_rate','spo2','temperature','resp_rate','bp') "
        "ORDER BY measured_at DESC, id DESC LIMIT 20", (uid,))
    recent = [{"metric": r["metric"], "value": r["value"], "value2": r["value2"],
               "measured_at": _iso(r["measured_at"]), "source": r["source"] or "manual"} for r in recent_rows]
    tokens = query_all("SELECT id, label, token_prefix, last_seen, created_at FROM twin_tokens WHERE user_id = ? ORDER BY id DESC", (uid,))
    return {"latest": latest, "history": history, "recent": recent, "tokens": tokens}


@app.route("/api/twin/state")
@login_required
def api_twin_state():
    try:
        days = int(request.args.get("days", "1"))
    except ValueError:
        days = 1
    days = days if days in (1, 7, 30) else 1
    return jsonify(_twin_snapshot(g.user["id"], days))


@app.route("/api/twin/ble-reading", methods=["POST"])
@login_required
def api_twin_ble_reading():
    if not rate_ok("twin_ble", 360):
        return jsonify(error="Too many BLE saves this hour."), 429
    b = request.get_json(silent=True) or {}
    ok, err = _save_sensor_reading(g.user["id"], b.get("metric"), b.get("value"), b.get("value2"), b.get("measured_at"), "ble")
    if not ok:
        return jsonify(error=err), 400
    _twin_notify(g.user["id"])
    return jsonify(ok=True)


@app.route("/api/twin/tokens", methods=["POST"])
@login_required
def api_twin_token_create():
    b = request.get_json(silent=True) or {}
    label = str(b.get("label", "")).strip()[:60] or "ESP32 Node"
    uid = g.user["id"]
    if not rate_ok("twin_tok", 20):
        return jsonify(error="Too many token changes. Please try again later."), 429
    if query("SELECT COUNT(*) AS n FROM twin_tokens WHERE user_id = ?", (uid,), one=True)["n"] >= 10:
        return jsonify(error="Limit of 10 ESP32 tokens reached. Revoke an unused one first."), 400
    raw = "mb_esp32_" + secrets.token_urlsafe(24)
    thash = hashlib.sha256(raw.encode()).hexdigest()
    query("INSERT INTO twin_tokens (user_id, label, token_hash, token_prefix, last_seen, created_at) VALUES (?,?,?,?,?,?)",
          (uid, label, thash, raw[:14], "", _utcnow()), commit=True)
    audit("twin_token_created", label)
    tokens = query_all("SELECT id, label, token_prefix, last_seen, created_at FROM twin_tokens WHERE user_id = ? ORDER BY id DESC", (uid,))
    return jsonify(token=raw, tokens=tokens)


@app.route("/api/twin/tokens/<int:tid>/delete", methods=["POST"])
@login_required
def api_twin_token_delete(tid):
    query("DELETE FROM twin_tokens WHERE id = ? AND user_id = ?", (tid, g.user["id"]), commit=True)
    audit("twin_token_revoked")
    tokens = query_all("SELECT id, label, token_prefix, last_seen, created_at FROM twin_tokens WHERE user_id = ? ORDER BY id DESC", (g.user["id"],))
    return jsonify(tokens=tokens)


@app.route("/api/twin/ingest", methods=["POST"])
def api_twin_ingest():
    """ESP32 HTTP ingestion endpoint (Bearer device token OR logged-in browser session)."""
    uid, tok_id = None, None
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        raw_tok = auth[7:].strip()
        if raw_tok:
            thash = hashlib.sha256(raw_tok.encode()).hexdigest()
            row = query("SELECT id, user_id FROM twin_tokens WHERE token_hash = ?", (thash,), one=True)
            if row:
                uid, tok_id = row["user_id"], row["id"]
    elif g.user:
        uid = g.user["id"]
    if uid is None:
        return jsonify(error="Unauthorized. Provide a valid ESP32 Bearer token."), 401
    if not _hit("twin_ingest", uid, 720, 3600):
        return jsonify(error="Ingestion rate limit reached."), 429

    b = request.get_json(silent=True) or {}
    dev_note = str(b.get("device_name", "ESP32")).strip()[:80]
    saved = []
    mapping = [("heart_rate", b.get("hr", b.get("heart_rate"))),
               ("spo2", b.get("spo2")),
               ("temperature", b.get("temp", b.get("temperature"))),
               ("resp_rate", b.get("rr", b.get("resp_rate")))]
    for metric, val in mapping:
        if val is not None:
            ok, _ = _save_sensor_reading(uid, metric, val, None, b.get("measured_at"), "esp32", dev_note)
            if ok:
                saved.append(metric)
    if b.get("bp_sys") is not None and b.get("bp_dia") is not None:
        ok, _ = _save_sensor_reading(uid, "bp", b.get("bp_sys"), b.get("bp_dia"), b.get("measured_at"), "esp32", dev_note)
        if ok:
            saved.append("bp")
    if not saved:
        return jsonify(error="No valid sensor values found (supported: hr, spo2, temp, rr, bp_sys/bp_dia)."), 400
    if tok_id is not None:
        query("UPDATE twin_tokens SET last_seen = ? WHERE id = ?", (_utcnow(), tok_id), commit=True)
    _twin_notify(uid)
    return jsonify(ok=True, saved=saved)


@app.route("/api/twin/stream")
@login_required
def api_twin_stream():
    uid = g.user["id"]
    q = queue.Queue(maxsize=20)
    with _TWIN_LOCK:
        _TWIN_SUBS.setdefault(uid, []).append(q)

    def gen():
        try:
            yield "event: twin_ready\ndata: {\"ok\": true}\n\n"
            while True:
                try:
                    msg = q.get(timeout=20)
                    yield "event: twin_update\ndata: " + json.dumps(msg) + "\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
     