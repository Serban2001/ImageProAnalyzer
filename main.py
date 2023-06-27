# This is a sample Python script.
import numpy as np
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from PyQt6.QtQuick import QQuickWindow

QQuickWindow.setSceneGraphBackend('software')

from PyQt6.QtWidgets import QApplication
from image_selector import ImageSelectorWindow

if __name__ == "__main__":
    app = QApplication([])
    window = ImageSelectorWindow()
    window.show()
    app.exec()
