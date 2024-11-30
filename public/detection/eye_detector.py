import cv2

# Load the Haar Cascade files for face and eye detection
face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')

def detect_eyes(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around the eyes
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return image

# Load an image and apply the eye detection
image = cv2.imread('path_to_image.jpg')
result_image = detect_eyes(image)

# Show the result
cv2.imshow('Detected Eyes', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
