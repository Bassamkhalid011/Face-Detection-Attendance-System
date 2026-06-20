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

### 2. Set up Database Credentials

Copy `.env.example` to `.env`:

```cmd
copy .env.example .env
```

Then open `.env` and fill in your Oracle DB details:

```
DB_USER=your_username
DB_PASSWORD=your_password
DB_DSN=localhost/ORCL
```

> **Note:** Never share your `.env` file. It is already listed in `.gitignore` so it will not be pushed to GitHub.

---

### 3. Set up Oracle Database

Make sure Oracle Database (XE or Standard) is running on your machine.
The system will create the `attendance` table automatically on first run.

---

### 4. Add student photos

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

---

### 5. Encode faces

Run this once (and again whenever you add or remove students):

```cmd
python encode_faces.py
```

This generates `encodings.pkl` which the recognition system uses.

---

### 6. Run the app

```cmd
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

### 6b. Run without GUI (command-line)

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

## File Structure

```
Face-Detection-Attendance-System-/
│
├── app.py                  → Streamlit GUI (main entry point)
├── main.py                 → Command-line camera + recognition loop
├── encode_faces.py         → Reads known_faces/, generates encodings.pkl
├── database.py             → Oracle DB connection and all queries
├── requirements.txt        → Python dependencies
├── .env                    → Your DB credentials (never share this)
├── .env.example            → Template for DB credentials
├── .gitignore              → Ensures .env is not pushed to GitHub
├── encodings.pkl           → Auto-generated — stores face encoding vectors
│
└── known_faces/            → Put student photo folders here
    ├── Ali/
    │   └── photo1.jpg
    └── Sara/
        └── photo1.jpg
```
