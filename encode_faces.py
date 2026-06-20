import face_recognition
import os
import pickle

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE  = "encodings.pkl"


def encode_all_faces():
    known_encodings = []
    known_names     = []

    print("[ENCODE] Starting face encoding...")

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_folder = os.path.join(KNOWN_FACES_DIR, person_name)

        if not os.path.isdir(person_folder):
            continue

        print(f"[ENCODE] Processing: {person_name}")

        for filename in os.listdir(person_folder):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_folder, filename)
            image    = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) == 0:
                print(f"  [WARN] No face found in {filename} — skipping.")
                continue

            known_encodings.append(encodings[0])
            known_names.append(person_name)
            print(f"  [OK] Encoded {filename}")

    # Save to file so main.py can load fast
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    print(f"\n[ENCODE] Done! {len(known_names)} face(s) encoded → saved to {ENCODINGS_FILE}")
    return known_encodings, known_names


if __name__ == "__main__":
    encode_all_faces()
