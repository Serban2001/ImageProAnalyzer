from PyQt6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget, QLineEdit
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from kmeans_processor import KMeansProcessor
from knn_labeled_processor import KNNProcessor
from contour_segmentation import ContourProcessor
from grab_cut_processor import GrabCutProcessor
from isodata_processor import IsoDataProcessor


class ImageSelectorWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create a label to display the image
        self.image_path = None
        self.image_label = QLabel()
        self.image_label.setFixedSize(300, 300)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Create a button to open the file dialog
        self.open_button = QPushButton("Open Image")
        self.open_button.clicked.connect(self.open_image)

        # Create additional widgets (initially hidden)
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter number of clusters")
        self.input_field.setText("3")  # Set the default value of clusters
        self.input_field.hide()

        # Create additional buttons (initially hidden)
        self.kmeans_button = QPushButton("K-means")
        self.kmeans_button.clicked.connect(self.process_kmeans)
        self.kmeans_button.hide()

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset)
        self.reset_button.hide()

        self.knn_button = QPushButton("KNN with labels")
        self.knn_button.clicked.connect(self.process_knn)
        self.knn_button.hide()

        self.contour_button = QPushButton("Contour detection")
        self.contour_button.clicked.connect(self.contour_detection)
        self.contour_button.hide()

        self.grab_cut_button = QPushButton("GrabCut")
        self.grab_cut_button.clicked.connect(self.process_grab_cut)
        self.grab_cut_button.hide()

        self.isodata_button = QPushButton("Isodata")
        self.isodata_button.clicked.connect(self.process_isodata)
        self.isodata_button.hide()

        # Create a vertical layout and add the widgets to it
        layout = QVBoxLayout()
        layout.addWidget(self.input_field)
        layout.addWidget(self.image_label)
        layout.addWidget(self.open_button)
        layout.addWidget(self.kmeans_button)
        layout.addWidget(self.knn_button)
        layout.addWidget(self.contour_button)
        layout.addWidget(self.grab_cut_button)
        layout.addWidget(self.isodata_button)
        layout.addWidget(self.reset_button)

        # Create a central widget and set the layout on it
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def open_image(self):
        # Open the file dialog to select an image
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Select Image")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.jpeg *.bmp)")

        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            # Get the selected image file path
            selected_files = file_dialog.selectedFiles()
            image_path = selected_files[0]

            # Display the selected image
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))

            # Show additional buttons
            self.input_field.show()
            self.kmeans_button.show()
            self.knn_button.show()
            self.contour_button.show()
            self.grab_cut_button.show()
            self.isodata_button.show()
            self.reset_button.show()
            self.open_button.hide()
            # Store the image path
            self.image_path = image_path

    def process_kmeans(self):
        value = self.input_field.text()
        KMeansProcessor.processImage(self.image_path, value)

    def process_knn(self):
        value = self.input_field.text()
        KNNProcessor.processImage(self.image_path, value)

    def contour_detection(self):
        ContourProcessor.processImage(self.image_path)

    def process_grab_cut(self):
        GrabCutProcessor.processImage(self.image_path)

    def process_isodata(self):
        value = self.input_field.text()
        IsoDataProcessor.processImage(self.image_path, value)

    def reset(self):
        # Reset the window to its initial state
        self.image_label.clear()
        self.open_button.show()
        self.kmeans_button.hide()
        self.reset_button.hide()
        self.input_field.hide()
        self.knn_button.hide()
        self.isodata_button.hide()
        self.contour_button.hide()
        self.grab_cut_button.hide()
        self.grab_cut_button.hide()
