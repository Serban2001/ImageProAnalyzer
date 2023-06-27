import cv2
import numpy as np


class IsoDataProcessor:
    @staticmethod
    def processImage(image_path, value):
        # Perform the image processing logic here
        # Use the provided image path and value
        print("Processing image:", image_path)
        print("Clusters Value:", value)

        image = cv2.imread(image_path)
        # Convert the image to RGB color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        max_iterations = 10
        min_cluster_size = 100
        max_cluster_size = 1000
        merge_threshold = 20
        K = int(value)

        # Reshape the image to a 2D array of pixels
        pixels = image.reshape(-1, 3).astype(np.float32)

        # Initialize clusters using K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Perform ISODATA algorithm
        iteration = 0
        while iteration < max_iterations:
            # Calculate cluster statistics
            cluster_sizes = np.bincount(labels.flatten())
            cluster_means = centers
            cluster_stddevs = np.zeros((K, 3), np.float32)  # Initialize std deviations array

            for i in range(K):
                cluster_pixels = pixels[labels.flatten() == i]
                if len(cluster_pixels) > 0:
                    cluster_stddevs[i] = np.std(cluster_pixels, axis=0)

            # Merge clusters
            for i in range(K):
                if cluster_sizes[i] < min_cluster_size:
                    closest_cluster = np.argmin(np.linalg.norm(centers - centers[i], axis=1))
                    labels[labels == i] = closest_cluster

            # Split clusters
            for i in range(K):
                if cluster_sizes[i] > max_cluster_size:
                    new_cluster_center = centers[i] + cluster_stddevs[i] / 2
                    centers = np.vstack((centers, new_cluster_center))
                    K += 1
                    labels[labels == i] = K - 1

            # Reassign pixels to the nearest cluster
            _, labels, centers = cv2.kmeans(pixels, K, labels, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            # Check convergence
            if iteration > 0:
                if np.max(np.abs(previous_labels - labels)) == 0:
                    break

            previous_labels = np.copy(labels)
            iteration += 1

        # Reshape the labels to the original image shape
        segmented_image = labels.reshape(image.shape[:2])

        # Display the results
        cv2.imshow('Input Image', image)
        cv2.imshow('Isodata Result', segmented_image.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ISODATA (Iterative Self-Organizing Data Analysis Technique) algorithm is an unsupervised clustering algorithm
# commonly used for image segmentation. It is an extension of the K-means algorithm and is often applied to partition
# an image into distinct regions or objects based on pixel intensities. Here's a step-by-step explanation of the
# ISODATA algorithm for image segmentation:
#
# Initialize parameters: Set initial values for parameters such as the desired number of clusters (K), the maximum
# number of iterations, minimum and maximum cluster sizes, and the threshold for cluster merging.
#
# Initialize clusters: Randomly assign each pixel in the image to one of the K initial clusters. You can use the
# K-means algorithm for this step.
#
# Calculate cluster statistics: Calculate the mean, standard deviation, and number of pixels in each cluster.
#
# Iterative process: a. Merge clusters: If the number of clusters is greater than the desired number (K),
# check if any clusters have a size below the minimum cluster size threshold. If so, merge them with their closest
# neighboring cluster based on some distance metric (e.g., Euclidean distance between cluster means).
#
# b. Split clusters: If the number of clusters is less than the desired number (K) and there are clusters with sizes
# above the maximum cluster size threshold, split them into two new clusters based on some criterion (e.g.,
# splitting along the highest standard deviation dimension).
#
# c. Update cluster statistics: Recalculate the mean, standard deviation, and number of pixels for each cluster after
# merging and splitting.
#
# d. Reassign pixels: Reassign each pixel to the cluster with the nearest mean intensity value.
#
# e. Check convergence: Check if the algorithm has reached the maximum number of iterations or if the changes in
# cluster assignments and cluster statistics are below a certain threshold. If not, go back to step 4a.
#
# Output: The final segmentation result is obtained after convergence, where each pixel is assigned to one of the K
# clusters.
