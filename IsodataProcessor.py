import cv2
import numpy as np


class IsoDataProcessorImage:
    @staticmethod
    def processImage(image_path,value):
        print("Processing image:", image_path)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        max_iterations = 10
        min_cluster_size = 100
        max_cluster_size = 1000
        K = value

        pixels = image.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        iteration = 0
        while iteration < max_iterations:
            cluster_sizes = np.bincount(labels.flatten())
            cluster_means = centers
            cluster_stddevs = np.zeros((K, 3), np.float32)

            for i in range(K):
                cluster_pixels = pixels[labels.flatten() == i]
                if len(cluster_pixels) > 0:
                    cluster_stddevs[i] = np.std(cluster_pixels, axis=0)

            for i in range(K):
                if cluster_sizes[i] < min_cluster_size:
                    closest_cluster = np.argmin(np.linalg.norm(centers - centers[i], axis=1))
                    labels[labels == i] = closest_cluster

            for i in range(K):
                if cluster_sizes[i] > max_cluster_size:
                    new_cluster_center = centers[i] + cluster_stddevs[i] / 2
                    centers = np.vstack((centers, new_cluster_center))
                    K += 1
                    labels[labels == i] = K - 1

            _, labels, centers = cv2.kmeans(pixels, K, labels, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            if iteration > 0:
                if np.max(np.abs(previous_labels - labels)) == 0:
                    break

            previous_labels = np.copy(labels)
            iteration += 1

        segmented_image = labels.reshape(image.shape[:2])

        cv2.imshow('Isodata Result', segmented_image.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
