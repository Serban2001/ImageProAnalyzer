import cv2
from PyQt6.QtCore import QTimer, QThread
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import  QLabel, QMainWindow, QWidget, QHBoxLayout
import logging as log

from GrabCutProcessor import GrabCutProcessorVideo
from KMeansProcessor import KMeansProcessorVideo
from KNNProcessor import KNNProcessorVideo
from FaceDetection import FaceDetectionProcessor
from ContourProcessor import ContourProcessorVideo
log.basicConfig(filename='webcam.log', level=log.INFO)



class CameraWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create a label to display the camera feed
        self.image_label = QLabel()
        self.setCentralWidget(self.image_label)
        self.resize(300, 300)

        # Create a timer to get the camera feed every so often
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # The processing type controls what processing is applied to each frame
        self.processing_type = 1

    def start_capture(self, width=1280, height=720):
        # Initialize webcam
        self.capture = cv2.VideoCapture(0)

        # Set the video width and height
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.timer.start(120)  # get a new frame every 30 ms

    def stop_capture(self):
        self.timer.stop()  # stop the timer
        self.capture.release()  # release the camera

    def closeEvent(self, event):
        self.stop_capture()
        event.accept()

    def set_processing_type(self, processing_type):
        self.processing_type = processing_type
        # Resetăm afișajul camerei atunci când schimbăm tipul de procesare
        if hasattr(self, 'capture'):
            self.update_frame()
    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            if self.processing_type == 1:
                frame = KMeansProcessorVideo.process_frame(frame)
            elif self.processing_type == 2:
                frame = KNNProcessorVideo.processImage(frame, 4)  # replace 4 with the clusters value you want
            elif self.processing_type == 3:
                print("ceva")
                frame = ContourProcessorVideo.process_frame(frame)
            elif self.processing_type == 4:
                frame = GrabCutProcessorVideo.process_frame(frame)
            elif self.processing_type == 5:
                frame = FaceDetectionProcessor.process_frame(frame)

            # Convert the frame to a QImage
            image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888)

            # Convert the QImage to a QPixmap and display it in the label
            self.image_label.setPixmap(QPixmap.fromImage(image))


