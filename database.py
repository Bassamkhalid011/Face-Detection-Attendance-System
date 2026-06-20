import oracledb
import os
from datetime import date, datetime
from dotenv import load_dotenv

load_dotenv()

DB_USER     = os.getenv("DB_USER", "system")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_DSN      = os.getenv("DB_DSN", "localhost/ORCL")


def connect():
    return oracledb.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        dsn=DB_DSN
    )


def create_tables():
    db = connect()
    cursor = db.cursor()

    # Check if table already exists
    cursor.execute("""
        SELECT COUNT(*) FROM user_tables WHERE table_name = 'ATTENDANCE'
    """)
    exists = cursor.fetchone()[0]

    if not exists:
        cursor.execute("""
            CREATE TABLE attendance (
                id           NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                student_name VARCHAR2(100),
                att_date     DATE,
                att_time     VARCHAR2(20),
                status       VARCHAR2(20) DEFAULT 'Present'
            )
        """)
        db.commit()
        print("[DB] Attendance table created.")
    else:
        print("[DB] Table already exists — ready.")

    db.close()


def already_marked(name):
    db = connect()
    cursor = db.cursor()

    cursor.execute("""
        SELECT id FROM attendance
        WHERE student_name = :name
        AND TRUNC(att_date) = TRUNC(SYSDATE)
    """, name=name)

    result = cursor.fetchone()
    db.close()
    return result is not None


def mark_attendance(name):
    if already_marked(name):
        print(f"[DB] {name} already marked today — skipping.")
        return False

    db = connect()
    cursor = db.cursor()

    now = datetime.now()
    cursor.execute("""
        INSERT INTO attendance (student_name, att_date, att_time)
        VALUES (:name, SYSDATE, :time)
    """, name=name, time=now.strftime("%H:%M:%S"))

    db.commit()
    db.close()
    print(f"[DB] Marked: {name} at {now.strftime('%H:%M:%S')}")
    return True


def get_today_attendance():
    db = connect()
    cursor = db.cursor()

    cursor.execute("""
        SELECT student_name, att_time, status
        FROM attendance
        WHERE TRUNC(att_date) = TRUNC(SYSDATE)
        ORDER BY att_time
    """)

    rows = cursor.fetchall()
    db.close()
    return rows


def get_attendance_by_date(target_date):
    """Return attendance records for a specific date (datetime.date object)."""
    db = connect()
    cursor = db.cursor()

    cursor.execute("""
        SELECT student_name, att_time, status
        FROM attendance
        WHERE TRUNC(att_date) = :d
        ORDER BY att_time
    """, d=target_date)

    rows = cursor.fetchall()
    db.close()
    return rows


def get_all_attendance():
    """Return all attendance records ordered by date and time."""
    db = connect()
    cursor = db.cursor()

    cursor.execute("""
        SELECT student_name,
               TO_CHAR(att_date, 'YYYY-MM-DD') AS att_date,
               att_time,
               status
        FROM attendance
        ORDER BY att_date DESC, att_time
    """)

    rows = cursor.fetchall()
    db.close()
    return rows


def get_student_list():
    """Return distinct student names that have been registered (have attendance)."""
    db = connect()
    cursor = db.cursor()

    cursor.execute("""
        SELECT DISTINCT student_name FROM attendance ORDER BY student_name
    """)

    rows = [r[0] for r in cursor.fetchall()]
    db.close()
    return rows


def delete_attendance_record(student_name, att_date, att_time):
    """Delete a specific attendance record."""
    db = connect()
    cursor = db.cursor()

    cursor.execute("""
        DELETE FROM attendance
        WHERE student_name = :name
          AND TO_CHAR(att_date, 'YYYY-MM-DD') = :d
          AND att_time = :t
    """, name=student_name, d=att_date, t=att_time)

    db.commit()
    db.close()


def test_connection():
    """Test DB connection. Returns (True, version_string) or (False, error_string)."""
    try:
        db = connect()
        cursor = db.cursor()
        cursor.execute("SELECT * FROM v$version WHERE ROWNUM = 1")
        version = cursor.fetchone()[0]
        db.close()
        return True, version
    except Exception as e:
        return False, str(e)
