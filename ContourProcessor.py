import cv2
import numpy as np

import cv2
import numpy as np

class ContourProcessorVideo:

    @staticmethod
    def process_frame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a binary threshold to the image
        _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize a black image with the same dimensions as the frame
        black_image = np.zeros_like(frame)

        # Draw each contour on the black image
        for contour in contours:
            cv2.drawContours(black_image, [contour], -1, (0, 255, 0), 2)

        # Use the black image with drawn contours as the output
        return black_image

class ContourProcessorImage:
    @staticmethod
    def processImage(image_path):
        # Image Segmentation using Contour Detection
        sample_image = cv2.imread(image_path)
        img = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
        cv2.imshow('Threshold', thresh)

        edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)
        cv2.imshow('Detected Edges', edges)

        cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
        mask = np.zeros((256, 256), np.uint8)
        masked = cv2.drawContours(mask, [cnt], -1, 255, -1)

        cv2.imshow('Masked', masked)

        dst = cv2.bitwise_and(img, img, mask=mask)
        segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

        cv2.imshow('Segmented', segmented)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
