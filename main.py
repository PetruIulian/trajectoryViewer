import cv2
import numpy as np
import time

camera = cv2.VideoCapture(0)

# Variabile pentru culoare selectată și traiectorie
selected_color = (0, 0, 0)  # Implicit negru
trajectory_points = []

color_threshold = 40
trail_duration = 1.5  # Durata trail-ului (secunde)
min_area = 300  # Aria minimă a unui contur pentru a-l considera valid

# Funcția de click pentru a selecta culoarea
def on_click(event, x, y, flags, param):
    global selected_color
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param
        if y < frame.shape[0] and x < frame.shape[1]:
            selected_color = tuple(int(c) for c in frame[y, x])
            print(f"Clicked at: ({x}, {y}), Color: {selected_color}")

cv2.namedWindow('Camera Feed')

while True:
    ret, frame = camera.read()
    if not ret:
        break

    now = time.time()

    cv2.setMouseCallback('Camera Feed', on_click, param=frame)

    # Convertim cadrul în HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_bgr = np.uint8([[selected_color]])
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]

    # Verificăm să nu avem negru sau alb, caz în care nu desenăm trail
    if selected_color == (0, 0, 0) or selected_color == (255, 255, 255):
        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Definim limitele culorii în HSV
    lower_bound = np.array([max(0, int(color_hsv[0]) - 10), 50, 50])
    upper_bound = np.array([min(179, int(color_hsv[0]) + 10), 255, 255])

    # Creăm masca pentru a extrage doar pixelii de culoare selectată
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Opțional: aplicăm un filtru Gaussian pentru a reduce zgomotul
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Alegem cel mai mare contur
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > min_area:
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Filtru pe distanța dintre punctele curente și cele anterioare
                if trajectory_points:
                    last_x, last_y, _ = trajectory_points[-1]
                    dist = np.hypot(cx - last_x, cy - last_y)
                    if dist < 50:  # Ignorăm salturile mari
                        trajectory_points.append((cx, cy, now))
                else:
                    trajectory_points.append((cx, cy, now))

    # Curățăm punctele care sunt mai vechi de 'trail_duration' secunde
    trajectory_points = [(x, y, t) for (x, y, t) in trajectory_points if now - t < trail_duration]

    # Desenăm traiectoria (doar ultimele puncte)
    for i in range(1, len(trajectory_points)):
        x1, y1, t1 = trajectory_points[i - 1]
        x2, y2, t2 = trajectory_points[i]
        age = now - t2
        alpha = max(0, 1 - age / trail_duration)
        color = (0, int(255 * alpha), int(255 * alpha))  # Culoare graduală, de la galben la transparent
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    cv2.rectangle(frame, (10, 10), (110, 60), selected_color, -1)
    cv2.putText(frame, str(selected_color), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

   
    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
