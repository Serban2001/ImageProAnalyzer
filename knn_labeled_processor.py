import cv2
import numpy as np


class KNNProcessor:
    @staticmethod
    def processImage(image_path, value):
        # Perform the image processing logic here
        # Use the provided image path and value
        print("Processing image:", image_path)
        print("Clusters Value:", value)

        sample_image = cv2.imread(image_path)
        img = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

        # start knn
        img_BGR = cv2.imread(image_path)
        if img_BGR is None:
            raise FileNotFoundError("'{0}' could not be opened!".format(image_path))

        BGR_COLORS = dict(blue=(255, 0, 0), green=(0, 255, 0), black=(0, 0, 0), white=(255, 255, 255))
        LABELS = dict(blue=np.array([0]), green=np.array([1]), black=np.array([3]), white=np.array([4]))
        trainData = np.array([BGR_COLORS['blue'], BGR_COLORS['green'], BGR_COLORS['black'], BGR_COLORS['white']],
                             dtype=np.float32)
        responses = np.array([LABELS['blue'], LABELS['green'], LABELS['black'], LABELS['white']], dtype=np.float32)

        knn = cv2.ml.KNearest_create()
        knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)

        img_vstacked = np.vstack(img_BGR).astype(np.float32)
        ret, results, neighbours, dist = knn.findNearest(img_vstacked, 1)

        height, width, depth = img_BGR.shape
        results_int = results.reshape(height, width).astype(np.uint8)

        def colorPixels(image, results, colorName):
            image[results[:, :] == LABELS[colorName]] = BGR_COLORS[colorName]

        img_clustered = img_BGR.copy()
        for colorName in BGR_COLORS.keys():
            colorPixels(img_clustered, results_int, colorName)

        cv2.imshow("Original vs. KNN", np.hstack((img_BGR, img_clustered)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
