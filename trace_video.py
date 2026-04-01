import cv2
import numpy as np

cap = cv2.VideoCapture("/home/guusje-schellekens/Downloads/flame4.mp4")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)

trajectory = []
frame_count = 0

writer = cv2.VideoWriter('/home/guusje-schellekens/Downloads/traced_flame_final.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         20, size)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_hsv is not None:
            print(f"HSV at ({x},{y}): {current_hsv[y, x]}")

current_hsv = None
cv2.namedWindow("Tracking")
cv2.setMouseCallback("Tracking", mouse_callback)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame.shape[0] > frame.shape[1]:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    #Skip first 10 frames to avoid false starts
    if frame_count < 10:
        writer.write(frame)
        cv2.imshow("Tracking", frame)
        cv2.waitKey(30)
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    current_hsv = hsv

    #Dark burgundy range for RGB(70,13,40)
    lower_red = np.array([160, 150, 40])
    upper_red = np.array([180, 255, 90])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    cv2.imshow("Red Mask", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 200:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                if trajectory:
                    #Only add point if it hasn't jumped too far:
                    last_x, last_y = trajectory[-1]
                    dist = np.sqrt((cx - last_x)**2 + (cy - last_y)**2)
                    if dist < 100:
                        trajectory.append((cx, cy))
                        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
                else:
                    #Only start trajectory near known starting position:
                    if abs(cx - 168) < 50 and abs(cy - 309) < 50:
                        trajectory.append((cx, cy))
                        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)

    #Draw trajectory
    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 255, 0), 3)

    writer.write(frame)
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()