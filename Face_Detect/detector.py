import time
import threading
import pyttsx3
from ultralytics import YOLO
import cv2

class Detector:
    def __init__(self, model_path, source, thresh=0.5, cooling=3.0):
        self.model = YOLO(model_path, task='detect')
        self.labels = self.model.names
        self.source = source
        self.thresh = thresh
        self.cooling = cooling

        # Voice config
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        for v in voices:
            if "female" in v.name.lower():
                self.engine.setProperty('voice', v.id)
        rate = self.engine.getProperty('rate')
        self.engine.setProperty('rate', int(rate * 0.75))  # 0.25 slower

        self.last_spoken = 0
        self.muted = False
        self.running = False
        self.cap = None

    def speak(self, text):
        if not self.muted:
            threading.Thread(target=lambda: (self.engine.say(text), self.engine.runAndWait())).start()

    def start(self):
        if str(self.source).isdigit():
            self.cap = cv2.VideoCapture(int(self.source))
        else:
            self.cap = cv2.VideoCapture(self.source)
        self.running = True

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def toggle_mute(self):
        self.muted = not self.muted

    def frame_generator(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model(frame, verbose=False)
            detections = results[0].boxes
            now = time.time()

            if len(detections) > 0:
                box = detections[0]
                conf = box.conf.item()
                if conf >= self.thresh:
                    cls = int(box.cls.item())
                    name = self.labels[cls]
                    if now - self.last_spoken >= self.cooling:
                        self.speak(name)
                        self.last_spoken = now

            ret, jpeg = cv2.imencode('.jpg', frame)
            yield jpeg.tobytes()
