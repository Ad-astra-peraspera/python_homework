import cv2
import numpy as np

# Load the face cascade file
face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')
nose_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_mcs_nose.xml')

# Check if the face cascade file has been loaded
if face_cascade.empty():
	raise IOError('Unable to load the face cascade classifier xml file')

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Define the scaling factor
scaling_factor = 0.5

# Loop until you hit the Esc key
while True:
    # Capture the current frame and resize it
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor,
            interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run the face detector on the grayscale image
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles on the image
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        noses = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in noses:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)
            break
    # Display the image
    cv2.imshow('Face Nose Detector', frame)

    # Check if Esc key has been pressed
    c = cv2.waitKey(1)
    if c == 27:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
