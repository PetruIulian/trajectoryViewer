import cv2

camera = cv2.VideoCapture(0)

frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Variabilă globală pentru culoarea selectată
selected_color = (0, 0, 0)  # implicit: negru

def on_click(event, x, y, flags, param):
    global selected_color
    if event == cv2.EVENT_LBUTTONDOWN:
        # Extrage cadrul curent din param
        frame = param
        if y < frame.shape[0] and x < frame.shape[1]:
            selected_color = tuple(int(c) for c in frame[y, x])
            print(f"Clicked at: ({x}, {y}), Color: {selected_color}")

cv2.namedWindow('Camera Feed')

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Setează callback-ul mouse-ului cu cadrul actual
    cv2.setMouseCallback('Camera Feed', on_click, param=frame)

    # Desenează un dreptunghi în colțul din stânga sus cu culoarea selectată
    cv2.rectangle(frame, (10, 10), (110, 60), selected_color, -1)
    cv2.putText(frame, str(selected_color), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
