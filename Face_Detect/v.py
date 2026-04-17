import cv2

for i in range(6):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            print(f"Camera found at index {i}")
        cap.release()
