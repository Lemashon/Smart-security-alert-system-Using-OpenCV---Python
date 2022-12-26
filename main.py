import cv2
import numpy as np
import face_recognition
import winsound

# Set the path to the alarm sound file
alarm_sound_file = 'alarm.wav'

imgIntruder = face_recognition.load_image_file('ImagesBasic/intruder.jpg')
imgIntruder = cv2.cvtColor(imgIntruder,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('ImagesBasic/Test-intruder.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgIntruder)[0]
encodeIntruder = face_recognition.face_encodings(imgIntruder)[0]
cv2.rectangle(imgIntruder,(faceLoc[3],faceLoc[0],faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0],faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeIntruder],encodeTest)
faceDis = face_recognition.face_distance([encodeIntruder],encodeTest)

# Check if the faces match
if results[0]:
  # Play the alarm sound
  winsound.PlaySound(alarm_sound_file, winsound.SND_ASYNC)
  # Display an "intruder alert" message
  cv2.putText(imgTest, "Intruder Alert!", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
else:
  # Display the face distance
  cv2.putText(imgTest, f'{round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Intruder', imgIntruder)
cv2.imshow('Test-Intruder', imgTest)
cv2.waitKey(0)
