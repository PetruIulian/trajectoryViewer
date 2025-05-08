import cv2

camera = cv2.VideoCapture(0)

frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Display the frame
    cv2.imshow('Camera Feed', frame)

    # Wait for 1 ms and check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break