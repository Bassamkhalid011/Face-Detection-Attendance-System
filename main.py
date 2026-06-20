import cv2
import face_recognition
import pickle
import numpy as np
from database import create_tables, mark_attendance, get_today_attendance

ENCODINGS_FILE = "encodings.pkl"
TOLERANCE      = 0.5   # Lower = stricter matching (0.4 - 0.6 recommended)
SCALE          = 0.5   # Resize frame for faster processing


def load_encodings():
    try:
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        print(f"[MAIN] Loaded {len(data['names'])} known face(s): {set(data['names'])}")
        return data["encodings"], data["names"]
    except FileNotFoundError:
        print("[ERROR] encodings.pkl not found. Run encode_faces.py first!")
        exit()


def run():
    # Setup
    create_tables()
    known_encodings, known_names = load_encodings()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("[MAIN] Camera started. Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Shrink frame for speed, convert BGR→RGB for face_recognition
        small   = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        rgb     = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Detect faces
        locations  = face_recognition.face_locations(rgb)
        encodings  = face_recognition.face_encodings(rgb, locations)

        for face_encoding, face_location in zip(encodings, locations):
            matches   = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)
            distances = face_recognition.face_distance(known_encodings, face_encoding)

            name  = "Unknown"
            color = (0, 0, 255)   # Red for unknown

            if len(distances) > 0:
                best_idx = np.argmin(distances)
                if matches[best_idx]:
                    name  = known_names[best_idx]
                    color = (0, 255, 0)   # Green for recognized
                    mark_attendance(name)

            # Scale locations back to original frame size
            top, right, bottom, left = [int(v / SCALE) for v in face_location]

            # Draw box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw name label
            cv2.rectangle(frame, (left, bottom - 28), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        # Show today's attendance count on screen
        today = get_today_attendance()
        cv2.putText(frame, f"Present today: {len(today)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Face Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print final attendance report
    print("\n===== TODAY'S ATTENDANCE =====")
    records = get_today_attendance()
    if records:
        for name, time, status in records:
            print(f"  {name:<20} {str(time):<12} {status}")
    else:
        print("  No attendance recorded today.")
    print("==============================")


if __name__ == "__main__":
    run()
