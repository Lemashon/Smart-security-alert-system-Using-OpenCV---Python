import cv2
import numpy as np
import face_recognition
import winsound

# Load the reference face encodings from a file or database
reference_face_encodings = []

# Set the path to the alarm sound file
alarm_sound_file = 'alarm.wav'

# Capture video from a webcam or other video input device
video_capture = cv2.VideoCapture(0)

while True:
  # Read the frame from the video capture device
  ret, frame = video_capture.read()

  # Convert the frame to RGB
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Find the face locations in the frame
  face_locations = face_recognition.face_locations(frame)

  # Extract the face encodings for each detected face
  face_encodings = face_recognition.face_encodings(frame, face_locations)

  # Loop through each detected face
  for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # Draw a rectangle around the face
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Compare the face encoding with the reference face encodings
    match = face_recognition.compare_faces(reference_face_encodings, face_encoding)

    # If a match is found, play the alarm sound and display an "intruder alert" message
    if any(match):
      winsound.PlaySound(alarm_sound_file, winsound.SND_ASYNC)
      cv2.putText(frame, "Intruder Alert!", (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

  # Display the video feed
  cv2.imshow('Video', frame)

  # Check if the user pressed the 'q' key to quit
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release the video capture device
video_capture.release()

# Close all windows
cv2.destroyAllWindows()
