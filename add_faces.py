import cv2
import pickle
import os
import numpy as np

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
faces_data = []
i = 0
name = input("Enter your name: ")

while True:
    ret, frame = video.read()

    if not ret:
        print("Error: Could not read frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = gray[y:y + h, x:x + w]  # Convert to grayscale
        resized_img = cv2.resize(crop_img, (50, 50)).flatten()

        # Ensure proper reshaping
        if resized_img.shape[0] == 2500:  # 50x50 image size
            faces_data.append(resized_img)
        else:
            print("Error: Incorrect dimensions for resized_img.")

        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

# Ensure proper reshaping of faces_data
if len(faces_data) == 100:
    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape(faces_data.shape[0], -1)

    # Load existing data or create new arrays if not available
    if os.path.exists('data/faces_data.pkl'):
        with open('data/faces_data.pkl', 'rb') as f:
            faces = pickle.load(f)
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
    else:
        faces = np.empty((0, 2500), dtype=int)  # Assuming 50x50 image size
        names = []

    # Check if dimensions match before appending
    if faces_data.shape[1] == 2500:  # 50x50 image size
        if len(names) == faces.shape[0]:
            faces = np.append(faces, faces_data, axis=0)
            names += [name] * faces_data.shape[0]

            # Save the updated faces and names arrays using pickle
            with open('data/faces_data.pkl', 'wb') as f:
                pickle.dump(faces, f)
            with open('data/names.pkl', 'wb') as f:
                pickle.dump(names, f)
        else:
            print("Error: Number of samples mismatch.")
    else:
        print("Error: Incorrect dimensions for faces_data.")
