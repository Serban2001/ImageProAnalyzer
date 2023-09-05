import cv2
import numpy as np



class GrabCutProcessorVideo:
    @staticmethod
    def process_frame(frame):
        # Resize frame to a smaller size
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        mask = np.zeros(small_frame.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Modify the rectangle to cover the entire frame
        height, width = small_frame.shape[:2]
        margin = min(height, width) // 10
        rect = (margin, margin, width - 2 * margin, height - 2 * margin)

        # Use 1 iteration for GrabCut instead of 5
        try:
            cv2.grabCut(small_frame, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
        except Exception as e:
            print("Exception caught during GrabCut:", e)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = small_frame * mask2[:, :, np.newaxis]

        # Resize the output back to original size
        try:
            img = cv2.resize(img, (frame.shape[1], frame.shape[0]))
        except Exception as e:
            print("Exception caught during resize:", e)

        return img


class GrabCutProcessorImage:
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
