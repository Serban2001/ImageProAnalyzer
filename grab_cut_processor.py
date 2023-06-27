import cv2
import numpy as np


class GrabCutProcessor:
    @staticmethod
    def processImage(image_path):
        # Perform the image processing logic here
        # Use the provided image path and value
        print("Processing image:", image_path)

        img = cv2.imread(image_path)
        img = cv2.resize(img, (256, 256))
        assert img is not None, "file could not be read, check with os.path.exists()"
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (73, 56, 110, 605)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, np.newaxis]
        # Display the result
        cv2.imshow('GrabCut result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
