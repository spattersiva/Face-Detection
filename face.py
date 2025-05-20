import os
import pickle
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 642)
cap.set(4, 482)

# Load encoded face data
file_path = 'encodefile.p'
if not os.path.exists(file_path):
    print("Error: encodefile.p not found.")
    exit(1)

with open(file_path, 'rb') as file:
    encodelistknownname = pickle.load(file)

encodelistknown, studname = encodelistknownname
print('Encode file loaded successfully.')

# Initialize attendance file
attendance_file = 'attendance.xlsx'
if not os.path.exists(attendance_file):
    attendance = pd.DataFrame(columns=["Name", "Date", "Time", "Dept", "College"])
    attendance.to_excel(attendance_file, index=False)
else:
    attendance = pd.read_excel(attendance_file)

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image from webcam.")
        continue

    # Convert the image for face recognition processing
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Reduce size for faster processing
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facecurframe = face_recognition.face_locations(imgs)
    encodecurframe = face_recognition.face_encodings(imgs, facecurframe)

    for encodeface, faceloc in zip(encodecurframe, facecurframe):
        matches = face_recognition.compare_faces(encodelistknown, encodeface)
        facedistance = face_recognition.face_distance(encodelistknown, encodeface)

        matchindex = np.argmin(facedistance)
        name = "Unknown Face"

        if matches[matchindex]:
            name = studname[matchindex]

        y11, x22, y22, x11 = faceloc
        y11, x22, y22, x11 = y11 * 4, x22 * 4, y22 * 4, x11 * 4

        cv2.putText(img, name + " presented", (x11, y11 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        today = datetime.now().strftime("%d-%m-%Y")
        if name != "Unknown Face" and not ((attendance["Name"] == name) & (attendance['Date'] == today)).any():
            new_entry = pd.DataFrame({
                "Name": [name], "Date": [today],
                "Time": [datetime.now().strftime("%I:%M %p")],
                "Dept": ["BCA"], "College": ["Kalasalingam"]
            })
            attendance = pd.concat([attendance, new_entry], ignore_index=True)

    cv2.imshow("Face Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save attendance and release resources
attendance.to_excel(attendance_file, index=False, engine='openpyxl')
print("Attendance saved successfully!")
cap.release()
cv2.destroyAllWindows()
