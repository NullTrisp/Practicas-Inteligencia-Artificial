import cv2

# Load the cascade classifier model from a file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the default camera using the VideoCapture() function
# You can specify the camera index if you have multiple cameras connected to your computer
capture = cv2.VideoCapture(0)

# Check if the camera was opened successfully
if not capture.isOpened():
    print('Failed to open camera')
else:
    # Continuously capture frames from the camera
    while True:
        # Capture a frame from the camera
        ret, frame = capture.read()

        # Check if the frame was captured successfully
        if not ret:
            print('Failed to capture frame')
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray)

        # Draw a rectangle around each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                frame,
                "PERUANO DETECTAD0!",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0, 1),
                2)

        # Display the output frame
        cv2.imshow('Camera', frame)

        # Wait for a key press to close the window
        key = cv2.waitKey(1)
        if key == 27:  # Press the Escape key to exit
            break

    # Release the camera and close all windows
    capture.release()
    cv2.destroyAllWindows()
