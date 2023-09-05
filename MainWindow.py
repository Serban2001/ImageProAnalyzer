import cv2
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QLabel, QPushButton, QMainWindow, QWidget, QHBoxLayout, QFileDialog
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

from CameraWindow import CameraWindow
from ContourProcessor import ContourProcessorImage
from GrabCutProcessor import GrabCutProcessorImage
from KMeansProcessor import KMeansProcessorImage
from KNNProcessor import KNNProcessorImage
from IsodataProcessor import IsoDataProcessorImage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.button_style_active = """
        QPushButton {
            background-color: #4CAF50; /* Green */
            border-radius: 5px;
            color: white;
            padding: 10px 20px;
            font-family: "serif";
            font-size: 15px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #45a049; /* Darker green on hover */
        }
        """

        self.button_style_inactive = """
        QPushButton {
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #f56b6b, stop: 1 #8b0000);
            border-radius: 5px;
            color: white;
            padding: 10px 20px;
            font-family: "serif";
            font-size: 15px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #ff8b8b, stop: 1 #7a0000);
        }
        """
        window_style = """
        QMainWindow {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #8b0000, stop: 1 #1a0000);
        }
        """
        self.setStyleSheet(window_style)

        # Create Image and Camera buttons and apply the style
        self.image_button = QPushButton("Image")
        self.camera_button = QPushButton("Camera")
        self.image_button.setStyleSheet(self.button_style_active)
        self.camera_button.setStyleSheet(self.button_style_inactive)

        self.image_button.clicked.connect(self.set_image_processing)
        self.camera_button.clicked.connect(self.set_camera_processing)

        # Horizontal layout for Image and Camera buttons
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.image_button)
        h_layout.addWidget(self.camera_button)

        # Add h_layout to the main QVBoxLayout
        layout = QVBoxLayout()
        layout.addLayout(h_layout)

        # Create buttons for each processing type and add to layout
        self.buttons = []
        for i in range(1, 6):
            button = QPushButton("")
            button.setStyleSheet(self.button_style_inactive)
            self.buttons.append(button)
            layout.addWidget(button)

        # Create a central widget and set the layout on it
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        self.camera_window = None

        self.image_modes = ["K-means Clustering", "KNN", "Contour Processing", "GrabCut", "Isodata"]
        self.camera_modes = ["K-means Clustering", "KNN", "Contour Processing", "GrabCut", "Face Detection"]
        self.set_image_processing()

    def show_camera(self, i):
        if self.camera_window is not None:
            self.camera_window.stop_capture()
            self.camera_window.close()
        self.camera_window = CameraWindow()
        self.camera_window.set_processing_type(i)  # Setează tipul de procesare în funcție de butonul apăsat
        self.camera_window.start_capture()
        self.camera_window.show()

    def closeEvent(self, event):
        if self.camera_window:
            self.camera_window.stop_capture()
            self.camera_window.close()

        event.accept()
    def open_image_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Select Image")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.jpeg *.bmp)")

        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            selected_files = file_dialog.selectedFiles()
            return selected_files[0]
        return None

    def set_image_processing(self):
        for btn, mode in zip(self.buttons, self.image_modes):
            btn.setText(mode)
        self.set_processing_functionality('image')
        self.image_button.setStyleSheet(self.button_style_active)
        self.camera_button.setStyleSheet(self.button_style_inactive)

    def set_camera_processing(self):
        for btn, mode in zip(self.buttons, self.camera_modes):
            btn.setText(mode)
        self.set_processing_functionality('camera')
        self.image_button.setStyleSheet(self.button_style_inactive)
        self.camera_button.setStyleSheet(self.button_style_active)

    def set_processing_functionality(self, mode):
        for btn in self.buttons:
            try:
                btn.clicked.disconnect()
            except TypeError:
                pass

        if mode == 'image':
            for i, btn in enumerate(self.buttons, 1):
                btn.clicked.connect(lambda checked, i=i: self.show_image_processing(i))
        else:
            for i, btn in enumerate(self.buttons, 1):
                btn.clicked.connect(lambda checked, i=i: self.show_camera(i))

    def show_image_processing(self, i):
        try:
            # Open the image file dialog and get the image path
            self.image_path = self.open_image_file()

            if not self.image_path:
                print("No image selected.")
                return

            if i == 1:
                value = 10
                print("Clusters: ", value)
                processed_img = KMeansProcessorImage.processImage(self.image_path, value)
            elif i == 2:
                value = 5
                print("Clusters: ", value)
                processed_img = KNNProcessorImage.processImage(self.image_path, value)
            elif i == 3:
                processed_img = ContourProcessorImage.processImage(self.image_path)
            elif i == 4:
                processed_img = GrabCutProcessorImage.processImage(self.image_path)
            elif i == 5:
                value = 3
                print("Clusters: ", 3)
                processed_img = IsoDataProcessorImage.processImage(self.image_path,value)
            else:
                print("Invalid mode selected.")
                return
            if processed_img is None:
                return
            height, width, channel = processed_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(processed_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)

            # Create a QLabel to display the QPixmap and make it an instance variable
            self.label = QLabel()
            self.label.setPixmap(pixmap)
            self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.label.setWindowTitle("Processed Image")
            self.label.show()
        except Exception as e:
            print(f"Error during image display: {e}")