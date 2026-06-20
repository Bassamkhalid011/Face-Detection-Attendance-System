"""
Streamlit GUI for Face Detection Attendance System
Run with: streamlit run app.py
"""

import os
import pickle
import shutil
import subprocess
import threading
import time
from datetime import date, datetime

import cv2
import face_recognition
import numpy as np
import streamlit as st

from database import (
    create_tables,
    delete_attendance_record,
    get_all_attendance,
    get_attendance_by_date,
    get_student_list,
    get_today_attendance,
    mark_attendance,
    test_connection,
)

ENCODINGS_FILE = "encodings.pkl"
KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.5
SCALE = 0.5

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Face Attendance System",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #313244;
    }
    .metric-card h2 { color: #cdd6f4; font-size: 2.2rem; margin: 0; }
    .metric-card p  { color: #a6adc8; margin: 4px 0 0; }
    .status-present { color: #a6e3a1; font-weight: bold; }
    .status-absent  { color: #f38ba8; font-weight: bold; }
    .sidebar-title  { font-size: 1.4rem; font-weight: 700; color: #cdd6f4; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ────────────────────────────────────────────────────

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "camera_thread" not in st.session_state:
    st.session_state.camera_thread = None
if "stop_camera" not in st.session_state:
    st.session_state.stop_camera = False
if "last_recognized" not in st.session_state:
    st.session_state.last_recognized = []
if "db_ready" not in st.session_state:
    st.session_state.db_ready = False

# ── Helpers ───────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def load_encodings():
    if not os.path.exists(ENCODINGS_FILE):
        return None, None
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]


def get_known_students():
    """Return list of folder names in known_faces/."""
    if not os.path.isdir(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
    return [
        d
        for d in os.listdir(KNOWN_FACES_DIR)
        if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))
    ]


def run_encoding():
    result = subprocess.run(
        ["python", "encode_faces.py"],
        capture_output=True,
        text=True,
    )
    return result.stdout, result.stderr


def process_frame_for_recognition(frame, known_encodings, known_names):
    """Detect and recognise faces in a single frame. Returns annotated frame + names."""
    small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    recognized = []
    for face_encoding, face_location in zip(encodings, locations):
        matches = face_recognition.compare_faces(
            known_encodings, face_encoding, tolerance=TOLERANCE
        )
        distances = face_recognition.face_distance(known_encodings, face_encoding)

        name = "Unknown"
        color = (0, 0, 255)

        if len(distances) > 0:
            best_idx = int(np.argmin(distances))
            if matches[best_idx]:
                name = known_names[best_idx]
                color = (0, 255, 0)
                mark_attendance(name)
                recognized.append(name)

        top, right, bottom, left = [int(v / SCALE) for v in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 28), (right, bottom), color, cv2.FILLED)
        cv2.putText(
            frame,
            name,
            (left + 6, bottom - 8),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    return frame, recognized


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="sidebar-title">📷 Face Attendance</p>', unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Dashboard",
            "📸 Live Recognition",
            "👥 Manage Students",
            "📋 Attendance Records",
            "⚙️ Settings",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Quick DB status
    ok, info = test_connection()
    if ok:
        st.success("✅ Database Connected")
        if not st.session_state.db_ready:
            try:
                create_tables()
                st.session_state.db_ready = True
            except Exception:
                pass
    else:
        st.error("❌ DB Disconnected")
        st.caption(info[:120])

    st.divider()
    st.caption("Face Detection Attendance System")
    st.caption("Powered by Oracle DB + OpenCV")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Dashboard":
    st.title("🏠 Dashboard")
    st.markdown("Real-time overview of today's attendance.")

    today_records = []
    if st.session_state.db_ready:
        try:
            today_records = get_today_attendance()
        except Exception as e:
            st.error(f"Database error: {e}")

    total_students = len(get_known_students())
    present_count = len(today_records)
    absent_count = max(0, total_students - present_count)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""<div class="metric-card">
            <h2>{total_students}</h2><p>Total Students</p></div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card">
            <h2 style="color:#a6e3a1">{present_count}</h2><p>Present Today</p></div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="metric-card">
            <h2 style="color:#f38ba8">{absent_count}</h2><p>Absent / Not Recorded</p></div>""",
            unsafe_allow_html=True,
        )

    st.divider()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Today's Attendance")
        if today_records:
            import pandas as pd

            df = pd.DataFrame(today_records, columns=["Student", "Time", "Status"])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No attendance recorded yet for today.")

    with col_right:
        st.subheader("Quick Actions")
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.rerun()

        st.markdown("---")
        st.markdown(f"**Date:** {date.today().strftime('%B %d, %Y')}")
        st.markdown(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")

        enc_exists = os.path.exists(ENCODINGS_FILE)
        st.markdown(
            f"**Encodings:** {'✅ Ready' if enc_exists else '❌ Missing'}"
        )
        st.markdown(f"**Registered Students:** {total_students}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE RECOGNITION
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📸 Live Recognition":
    st.title("📸 Live Face Recognition")
    st.markdown(
        "The camera feed runs below. Recognised faces are automatically logged "
        "to the Oracle database (once per day per student)."
    )

    if not st.session_state.db_ready:
        st.warning("Database not connected. Please check Settings.")
        st.stop()

    if not os.path.exists(ENCODINGS_FILE):
        st.error(
            "No face encodings found (`encodings.pkl`). "
            "Go to **Manage Students** and click **Re-encode All Faces**."
        )
        st.stop()

    known_encodings, known_names = load_encodings()
    if known_encodings is None:
        st.error("Failed to load encodings.")
        st.stop()

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    start_btn = col_btn1.button("▶️ Start Camera", type="primary")
    stop_btn = col_btn2.button("⏹️ Stop Camera")

    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    recognized_placeholder = st.empty()

    if start_btn:
        st.session_state.camera_running = True
        st.session_state.stop_camera = False

    if stop_btn:
        st.session_state.camera_running = False
        st.session_state.stop_camera = True

    if st.session_state.camera_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open camera. Make sure a webcam is connected.")
            st.session_state.camera_running = False
        else:
            status_placeholder.success("🎥 Camera is running — recognised faces are logged automatically.")

            seen_today = set(n for n, _, _ in get_today_attendance())

            while st.session_state.camera_running and not st.session_state.stop_camera:
                ret, frame = cap.read()
                if not ret:
                    status_placeholder.error("Failed to read frame from camera.")
                    break

                annotated, newly_recognized = process_frame_for_recognition(
                    frame, known_encodings, known_names
                )
                seen_today.update(newly_recognized)

                rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                if seen_today:
                    recognized_placeholder.success(
                        f"✅ Marked today: {', '.join(sorted(seen_today))}"
                    )

                time.sleep(0.03)

            cap.release()
            frame_placeholder.empty()
            status_placeholder.info("Camera stopped.")
    else:
        frame_placeholder.markdown(
            """
            <div style="background:#1e1e2e;border-radius:12px;padding:80px;
                        text-align:center;border:1px solid #313244;">
                <span style="font-size:4rem">📷</span>
                <p style="color:#a6adc8;font-size:1.2rem;margin-top:16px;">
                    Press <strong>Start Camera</strong> to begin live recognition
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Snapshot mode
    st.divider()
    st.subheader("📷 Snapshot Recognition")
    st.markdown("Upload a photo to check attendance from an image.")
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded:
        import numpy as np
        from PIL import Image

        img = Image.open(uploaded).convert("RGB")
        img_np = np.array(img)
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        annotated, recognized = process_frame_for_recognition(bgr, known_encodings, known_names)
        rgb_out = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(rgb_out, caption="Recognition Result", use_container_width=True)
        if recognized:
            st.success(f"Recognised & logged: {', '.join(recognized)}")
        else:
            st.warning("No known faces detected in this image.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MANAGE STUDENTS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "👥 Manage Students":
    st.title("👥 Manage Students")

    students = get_known_students()

    col_add, col_list = st.columns([1, 1])

    with col_add:
        st.subheader("➕ Add New Student")
        new_name = st.text_input("Student Name", placeholder="e.g. Ali Hassan")
        photos = st.file_uploader(
            "Upload 3–5 clear face photos",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        if st.button("Save Student", type="primary"):
            if not new_name.strip():
                st.error("Please enter a student name.")
            elif not photos:
                st.error("Please upload at least one photo.")
            else:
                folder = os.path.join(KNOWN_FACES_DIR, new_name.strip())
                os.makedirs(folder, exist_ok=True)
                for photo in photos:
                    path = os.path.join(folder, photo.name)
                    with open(path, "wb") as f:
                        f.write(photo.read())
                st.success(
                    f"✅ Saved {len(photos)} photo(s) for **{new_name.strip()}**. "
                    "Click **Re-encode All Faces** below to apply."
                )
                st.rerun()

    with col_list:
        st.subheader("📋 Registered Students")
        if students:
            for s in sorted(students):
                folder = os.path.join(KNOWN_FACES_DIR, s)
                photo_count = len(
                    [
                        f
                        for f in os.listdir(folder)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                )
                with st.expander(f"👤 {s}  ({photo_count} photo{'s' if photo_count != 1 else ''})"):
                    # Show thumbnails
                    imgs = [
                        f
                        for f in os.listdir(folder)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                    if imgs:
                        thumb_cols = st.columns(min(len(imgs), 3))
                        for i, img_name in enumerate(imgs[:3]):
                            thumb_cols[i].image(
                                os.path.join(folder, img_name),
                                caption=img_name,
                                use_container_width=True,
                            )
                    if st.button(f"🗑️ Remove {s}", key=f"del_{s}"):
                        shutil.rmtree(folder)
                        st.warning(f"Removed student **{s}**. Re-encode faces to update.")
                        st.rerun()
        else:
            st.info("No students registered yet. Add one on the left.")

    st.divider()
    st.subheader("🔄 Re-encode All Faces")
    st.markdown(
        "Run this after adding or removing students to update the recognition model."
    )
    if st.button("▶️ Run encode_faces.py", type="primary"):
        with st.spinner("Encoding faces — this may take a minute…"):
            stdout, stderr = run_encoding()
        if stderr and "error" in stderr.lower():
            st.error(f"Encoding failed:\n```\n{stderr}\n```")
        else:
            st.success("✅ Encoding complete!")
            st.code(stdout)
        load_encodings.clear()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ATTENDANCE RECORDS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📋 Attendance Records":
    st.title("📋 Attendance Records")

    if not st.session_state.db_ready:
        st.warning("Database not connected. Please check Settings.")
        st.stop()

    import pandas as pd

    tab_today, tab_history, tab_search = st.tabs(
        ["📅 Today", "🗂️ Full History", "🔍 Search by Date"]
    )

    with tab_today:
        try:
            records = get_today_attendance()
        except Exception as e:
            st.error(f"DB error: {e}")
            records = []

        if records:
            df = pd.DataFrame(records, columns=["Student", "Time", "Status"])
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Export Today's CSV",
                csv,
                f"attendance_{date.today()}.csv",
                "text/csv",
            )
        else:
            st.info("No attendance recorded today.")

    with tab_history:
        try:
            all_records = get_all_attendance()
        except Exception as e:
            st.error(f"DB error: {e}")
            all_records = []

        if all_records:
            df_all = pd.DataFrame(all_records, columns=["Student", "Date", "Time", "Status"])
            st.dataframe(df_all, use_container_width=True, hide_index=True)

            csv_all = df_all.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Export Full History CSV",
                csv_all,
                "attendance_history.csv",
                "text/csv",
            )

            # Summary chart
            st.subheader("Attendance per Day")
            chart_data = df_all.groupby("Date").size().reset_index(name="Count")
            st.bar_chart(chart_data.set_index("Date"))
        else:
            st.info("No attendance records found in the database.")

    with tab_search:
        search_date = st.date_input("Select date", value=date.today())
        if st.button("🔍 Fetch Records"):
            try:
                recs = get_attendance_by_date(search_date)
            except Exception as e:
                st.error(f"DB error: {e}")
                recs = []

            if recs:
                df_s = pd.DataFrame(recs, columns=["Student", "Time", "Status"])
                st.dataframe(df_s, use_container_width=True, hide_index=True)
                csv_s = df_s.to_csv(index=False).encode()
                st.download_button(
                    f"⬇️ Export {search_date} CSV",
                    csv_s,
                    f"attendance_{search_date}.csv",
                    "text/csv",
                )
            else:
                st.info(f"No records found for {search_date}.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "⚙️ Settings":
    st.title("⚙️ Settings & Configuration")

    st.subheader("🗄️ Database Connection")

    import database as db_module

    with st.form("db_settings"):
        db_user = st.text_input("Oracle Username", value=db_module.DB_USER)
        db_pass = st.text_input("Oracle Password", value=db_module.DB_PASSWORD, type="password")
        db_dsn = st.text_input(
            "DSN (host/service)",
            value=db_module.DB_DSN,
            help="e.g. localhost/ORCL or 192.168.1.10/XEPDB1",
        )
        save_db = st.form_submit_button("💾 Save & Test Connection", type="primary")

    if save_db:
        db_module.DB_USER = db_user
        db_module.DB_PASSWORD = db_pass
        db_module.DB_DSN = db_dsn
        ok, info = test_connection()
        if ok:
            st.success(f"✅ Connected! Oracle version:\n{info}")
            try:
                create_tables()
                st.session_state.db_ready = True
            except Exception as e:
                st.warning(f"Tables check failed: {e}")
        else:
            st.error(f"❌ Connection failed: {info}")

    st.divider()
    st.subheader("🔧 Recognition Settings")
    st.markdown(
        "These settings apply to the **Live Recognition** page "
        "(restart the camera for changes to take effect)."
    )

    import database  # re-import to access module-level TOLERANCE/SCALE in main

    tol = st.slider("Matching Tolerance", 0.3, 0.8, TOLERANCE, 0.05,
                    help="Lower = stricter. 0.5 is recommended.")
    scale = st.slider("Frame Scale", 0.25, 1.0, SCALE, 0.05,
                      help="Lower = faster but less accurate.")

    st.info(
        f"Current values: tolerance={tol}, scale={scale}. "
        "These are session-only. Edit `app.py` to persist them."
    )

    st.divider()
    st.subheader("ℹ️ System Info")

    ok, ver = test_connection()
    st.markdown(f"- **DB Status:** {'✅ Connected' if ok else '❌ Disconnected'}")
    if ok:
        st.markdown(f"- **Oracle:** {ver}")
    st.markdown(f"- **Encodings file:** {'✅ Exists' if os.path.exists(ENCODINGS_FILE) else '❌ Missing'}")
    st.markdown(f"- **Known students:** {len(get_known_students())}")
    st.markdown(f"- **Python:** {__import__('sys').version.split()[0]}")
    st.markdown(f"- **OpenCV:** {cv2.__version__}")
    st.markdown(f"- **face_recognition:** {face_recognition.__version__}")
