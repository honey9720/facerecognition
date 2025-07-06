from flask import Flask, render_template, request, jsonify
import face_recognition
import numpy as np
import cv2
import base64
import os

app = Flask(__name__)

# Load known faces
known_encodings = []
known_names = []
for filename in os.listdir("known_faces"):
    img = face_recognition.load_image_file(f"known_faces/{filename}")
    encoding = face_recognition.face_encodings(img)[0]
    known_encodings.append(encoding)
    known_names.append(os.path.splitext(filename)[0])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recognize", methods=["POST"])
def recognize():
    data = request.json
    image_data = base64.b64decode(data["image"].split(",")[1])
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    rgb_img = img[:, :, ::-1]
    faces = face_recognition.face_locations(rgb_img)
    encodings = face_recognition.face_encodings(rgb_img, faces)

    results = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = "Unknown"
        if True in matches:
            idx = np.argmin(face_recognition.face_distance(known_encodings, encoding))
            name = known_names[idx]
        results.append(name)

    return jsonify({"names": results})
