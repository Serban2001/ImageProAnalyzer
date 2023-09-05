import cv2
import logging as log
from datetime import datetime as dt

log.basicConfig(filename='webcam.log', level=log.INFO)
class FaceDetectionProcessor:
    anterior = 0
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    @staticmethod
    def process_frame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FaceDetectionProcessor.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Logging face info
        if FaceDetectionProcessor.anterior != len(faces):
            FaceDetectionProcessor.anterior = len(faces)
            log.info("faces: " + str(len(faces)) + " at " + str(dt.now()))

        return frame