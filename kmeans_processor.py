import cv2
import numpy as np


class KMeansProcessor:
    @staticmethod
    def processImage(image_path, value):
        # Perform the image processing logic here
        # Use the provided image path and value
        print("Processing image:", image_path)
        print("Clusters Value:", value)

        sample_image = cv2.imread(image_path)
        img = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

        twoDimage = img.reshape((-1, 3))
        twoDimage = np.float32(twoDimage)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = int(value)
        attempts = 10

        ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape(img.shape)

        cv2.imshow("Original vs. Kmeans Clustered", np.hstack((sample_image, result_image)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
