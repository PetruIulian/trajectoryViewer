import cv2
import numpy as np
import time
import math
import os
import subprocess

# Valori maxime/minime pentru Brio (verificabile cu v4l2-ctl --list-ctrls)
PAN_MIN, PAN_MAX = -36000, 36000
TILT_MIN, TILT_MAX = -36000, 36000
ZOOM_MIN, ZOOM_MAX = 100, 400

# Variabile inițiale
pan_val = 0
tilt_val = 0
zoom_val = 100

def set_camera_control(control, value):
    try:
        subprocess.run([
            "v4l2-ctl", "-d", "/dev/video0", "--set-ctrl", f"{control}={value}"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to set {control}: {e}")

def on_pan(val):
    real_val = PAN_MIN + val * (PAN_MAX - PAN_MIN) // 100
    set_camera_control("pan_absolute", real_val)

def on_saturation(val):
    real_val = val * 255 // 100
    set_camera_control("saturation", real_val)


def on_focus(val):
    real_val = val * 255 // 100  # Presupunem intervalul focusului e 0–255
    set_camera_control("focus_absolute", real_val)

# ========= INIȚIALIZARE =========
if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using :0.0')
    os.environ['DISPLAY'] = ':0.0'

camera = cv2.VideoCapture('/dev/video0')
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera.set(cv2.CAP_PROP_FPS, 60)


# ---------- Restul variabilelor ----------
selected_color = (0, 0, 0)
trail_duration = 5.0
min_area = 300
max_dist = 60

# Kalman
def create_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                     [0, 1, 0, 1],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kf

trackers = []

def on_click(event, x, y, flags, param):
    global selected_color
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param
        if y < frame.shape[0] and x < frame.shape[1]:
            selected_color = tuple(int(c) for c in frame[y, x])
            print(f"Clicked at: ({x}, {y}), Color: {selected_color}")

# ---------- UI și loop ----------
cv2.namedWindow('Camera Feed', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Camera Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.createTrackbar('Pan', 'Camera Feed', 50, 100, on_pan)
cv2.createTrackbar('Saturation', 'Camera Feed', 50, 100, on_saturation)
cv2.createTrackbar('Focus', 'Camera Feed', 0, 100, on_focus)

# Setare inițială
on_saturation(50)
on_pan(50)
set_camera_control("zoom_absolute", 100)
on_focus(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    now = time.time()
    cv2.setMouseCallback('Camera Feed', on_click, param=frame)

    filtered = cv2.medianBlur(frame, 5)
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

    if selected_color in [(0, 0, 0), (255, 255, 255)]:
        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    color_bgr = np.uint8([[selected_color]])
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]

    lower_bound = np.array([
        max(0, int(color_hsv[0]) - 5),
        max(50, int(color_hsv[1]) - 40),
        max(50, int(color_hsv[2]) - 40)
    ])
    upper_bound = np.array([
        min(179, int(color_hsv[0]) + 5),
        min(255, int(color_hsv[1]) + 40),
        min(255, int(color_hsv[2]) + 40)
    ])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_centroids = []

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                detected_centroids.append((cx, cy))
                cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

    # Kalman + Traiectorii
    for cx, cy in detected_centroids:
        matched = False
        for kf, trajectory in trackers:
            pred_x, pred_y = kf.predict()[:2].flatten()
            dist = math.hypot(cx - pred_x, cy - pred_y)
            if dist < max_dist:
                kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
                x, y = kf.statePost[:2].flatten()
                trajectory.append((int(x), int(y), now))
                matched = True
                break
        if not matched:
            kf = create_kalman()
            kf.statePre = np.array([[np.float32(cx)], [np.float32(cy)], [0.], [0.]], dtype=np.float32)
            kf.statePost = kf.statePre.copy()
            kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
            trackers.append((kf, [(cx, cy, now)]))

    for i in range(len(trackers)):
        kf, trajectory = trackers[i]
        if trajectory and now - trajectory[-1][2] > 0.5:
            # Dacă n-a fost corectat recent, ignorăm predicția
            continue
        pred = kf.predict()
        x, y = pred[:2].flatten()
        trajectory.append((int(x), int(y), now))

    for i in range(len(trackers)):
        kf, traj = trackers[i]
        trackers[i] = (kf, [(x, y, t) for (x, y, t) in traj if now - t < trail_duration])

    for _, trajectory in trackers:
        for i in range(1, len(trajectory)):
            x1, y1, t1 = trajectory[i - 1]
            x2, y2, t2 = trajectory[i]
            age = now - t2
            alpha = max(0, 1 - age / trail_duration)
            color = (0, int(255 * alpha), int(255 * alpha))
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    cv2.rectangle(frame, (10, 10), (110, 60), selected_color, -1)
    cv2.putText(frame, str(selected_color), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
