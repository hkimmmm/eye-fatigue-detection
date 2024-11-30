import cv2
import numpy as np
from imutils import face_utils
import dlib

def eye_aspect_ratio(eye):
    # Menghitung Eye Aspect Ratio (EAR)
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_fatigue(eyes):
    # Ambang batas EAR untuk mendeteksi kelelahan (biasanya antara 0.2 dan 0.3)
    EAR_THRESHOLD = 0.25
    for eye in eyes:
        ear = eye_aspect_ratio(eye)
        if ear < EAR_THRESHOLD:
            print("Kelelahan Terdeteksi!")
            return True  # Kelelahan terdeteksi
    return False

def get_eye_landmarks(gray, detector, predictor):
    # Deteksi wajah pada gambar grayscale
    faces = detector(gray)
    eyes_landmarks = []

    for face in faces:
        # Ambil landmark untuk wajah
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Ambil koordinat mata kiri dan kanan
        left_eye = landmarks[42:48]  # Landmark mata kiri
        right_eye = landmarks[36:42]  # Landmark mata kanan

        eyes_landmarks.append(left_eye)
        eyes_landmarks.append(right_eye)

    return eyes_landmarks

# Inisialisasi detektor wajah dlib dan prediktor landmark wajah
detector = dlib.get_frontal_face_detector()

# Periksa apakah model file ada
predictor_path = "models/shape_predictor_68_face_landmarks.dat"
try:
    predictor = dlib.shape_predictor(predictor_path)
    print(f"Model berhasil dimuat dari {predictor_path}")
except RuntimeError as e:
    print(f"Gagal memuat model: {e}")
    exit(1)

# Mulai menangkap video dari webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Tidak dapat mengakses webcam.")
    exit(1)

while True:
    # Tangkap setiap frame dari webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Gagal menangkap gambar.")
        break

    # Ubah frame menjadi grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Dapatkan landmark mata dari frame saat ini
    eyes = get_eye_landmarks(gray, detector, predictor)

    if not eyes:
        print("Tidak ada mata yang terdeteksi!")
    else:
        # Deteksi kelelahan berdasarkan EAR untuk setiap mata yang terdeteksi
        if detect_fatigue(eyes):
            cv2.putText(frame, "Kelelahan Terdeteksi!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Tidak Ada Kelelahan", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame dengan anotasi
    cv2.imshow("Deteksi Kelelahan", frame)

    # Hentikan loop jika pengguna menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan webcam dan tutup jendela OpenCV
cap.release()
cv2.destroyAllWindows()
