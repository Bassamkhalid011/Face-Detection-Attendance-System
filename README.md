# Face Recognition Attendance System

Automatic attendance tracking using live face detection via webcam. Recognised faces are logged to an **Oracle Database** with date and time. Includes a full **Streamlit GUI** for live recognition, student management, and attendance reports.

---

## Features

- Real-time face detection and recognition via webcam
- Automatic attendance logging to Oracle DB (once per day per student)
- Streamlit web GUI with dashboard, live camera, student manager, and reports
- Snapshot recognition from uploaded images
- CSV export of attendance records
- Attendance charts and history view

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.8+ | |
| Oracle Database | XE (Express Edition) is free — use `localhost/ORCL` or `localhost/XEPDB1` |
| Oracle Instant Client | Required by `oracledb` in thick mode (optional for thin mode) |
| Webcam | For live recognition |
| `cmake` + C++ build tools | Required to install `dlib` (needed by `face_recognition`) |

---

## Setup

### 1. Install dependencies

```cmd
pip install -r requirements.txt
```

> **Note:** Installing `face_recognition` requires `dlib`, which needs CMake and a C++ compiler.
> On Windows, install [CMake](https://cmake.org/download/) and [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first.

---

### 2. Set up Oracle Database

Make sure Oracle Database (XE or Standard) is running on your machine.
The system will create the `attendance` table automatically on first run.

Default connection settings (edit `database.py` if yours differ):

```DB_USER     = "your_username"
DB_PASSWORD = "your_password"
DB_DSN      = "localhost/ORCL"
```

You can also change these settings from the **Settings** page in the GUI without editing files.

---

### 3. Add student photos

Create a folder for each student inside `known_faces/`:

```
known_faces/
├── Ali/
│   ├── photo1.jpg
│   └── photo2.jpg
├── Sara/
│   └── photo1.jpg
```

- Folder name = student name (exactly as it will appear in attendance)
- Use **3–5 clear, well-lit photos** per student for best accuracy
- Supported formats: `.jpg`, `.jpeg`, `.png`

You can also upload photos directly through the **Manage Students** page in the GUI.

---

### 4. Encode faces

Run this once (and again whenever you add or remove students):

```cmd
python encode_faces.py
```

This generates `encodings.pkl` which the recognition system uses.
You can also trigger this from the **Manage Students** page in the GUI.

---

### 5. Run the GUI (recommended)

```cmd
streamlit run app.py
```

Open your browser at **http://localhost:8501**

The GUI includes:

| Page | What it does |
|------|-------------|
| 🏠 Dashboard | Today's attendance summary with metrics |
| 📸 Live Recognition | Webcam feed with real-time face detection and logging |
| 👥 Manage Students | Add/remove students and trigger re-encoding |
| 📋 Attendance Records | View, filter, and export attendance by date |
| ⚙️ Settings | Configure Oracle DB connection and recognition parameters |

---

### 5b. Run without GUI (command-line)

```cmd
python main.py
```

- Camera opens and scans faces live
- Press **Q** to quit and print today's attendance report to the terminal

---

## How it works

1. Faces in `known_faces/` are encoded into 128-dimensional vectors and saved to `encodings.pkl`
2. The webcam captures frames; each frame is scaled down for speed then scanned for faces
3. Detected faces are compared to known encodings using Euclidean distance
4. If a match is found (distance ≤ 0.5 tolerance): name shown in **green box** → logged to Oracle DB
5. Unknown faces shown in **red box** → not logged
6. Each student is logged only **once per day**

---

## Database table

The system creates this table automatically:

```sql
CREATE TABLE attendance (
    id           NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    student_name VARCHAR2(100),
    att_date     DATE,
    att_time     VARCHAR2(20),
    status       VARCHAR2(20) DEFAULT 'Present'
)
```

---

## File structure

| File / Folder | Purpose |
|---------------|---------|
| `app.py` | Streamlit GUI (main entry point) |
| `main.py` | Command-line camera + recognition loop |
| `encode_faces.py` | Reads `known_faces/`, generates `encodings.pkl` |
| `database.py` | Oracle DB connection and all queries |
| `requirements.txt` | Python dependencies |
| `encodings.pkl` | Auto-generated — stores face encoding vectors |
| `known_faces/` | Put student photo folders here |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `oracledb` connection error | Check Oracle service is running: `lsnrctl status` (Windows: check Services) |
| `encodings.pkl not found` | Run `python encode_faces.py` or use Manage Students → Re-encode |
| Camera not opening | Check another app isn't using the webcam; try changing device index to `1` in `main.py` |
| Face not recognized | Add more photos (3–5), ensure good lighting, try lowering tolerance to `0.45` |
| `dlib` install fails | Install CMake and Visual Studio C++ Build Tools, then retry `pip install dlib` |
| DSN format for Oracle 21c XE | Use `localhost/XEPDB1` instead of `localhost/ORCL` |
