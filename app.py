from flask import Flask, render_template, Response
import cv2
import pickle
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import os
import csv

app = Flask(__name__)
camera = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

COL_NAMES = ['NAME', 'TIME']

attendance_written = False
def gen_frames():
    global attendance_written
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = gray[y:y+h, x:x+w]  # Convert to grayscale
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)

            ts = datetime.now().strftime("%H:%M:%S")
            attendance = [str(output[0]), str(ts)]

            # Display welcome message with the detected name
            welcome_message = f"Welcome, {attendance}!"
            cv2.putText(frame, welcome_message, (x, y - 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Save attendance to a CSV file
            date = datetime.now().strftime("%d-%m-%Y")
            attendance_file_path = f"Attendance/Attendance_{date}.csv"
            exist = os.path.isfile(attendance_file_path)

            with open(attendance_file_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if exist:
                    writer.writerow(attendance)
                else:
                    continue
                    

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login')
def login():
    return render_template('login.html')



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
