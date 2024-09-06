# Import necessary libraries
import pandas as pd
import numpy as np
import time

# Function to read data from a CSV file and preprocess it
def read_data(file_path):
    # Read CSV file into a DataFrame
    df = pd.read_csv(file_path)
    # Replace empty cells with NaN and drop rows with NaN values
    df.replace(' ', np.nan, inplace=True)
    df.dropna(inplace=True)
    # Reset DataFrame index and create a 'Point' column for indexing
    df.reset_index(drop=True, inplace=True)
    df.index = df.index + 1
    df['Point'] = df.index

    # Extract data values and point labels
    data = df.drop(['Point'], axis=1).values
    points_labels = df['Point'].values

    return data, points_labels

# Function to calculate single linkage distance between two clusters
def single_linkage(cluster1, cluster2):
    max_distance = -np.inf
    for point1 in cluster1:
        for point2 in cluster2:
            # Calculate cosine similarity between points
            distance = cosine_similarity(point1, point2)
            if distance > max_distance:
                max_distance = distance

    return max_distance

# Function for hierarchical clustering using single linkage agglomerative technique
def hierarchical_clustering(data, k):
    clusters = [[point] for point in data]
    cluster_labels = [[i+1] for i in range(len(data))]  # Labels to track the original points

    iteration = 0
    while len(clusters) > k:
        max_distance = -np.inf
        merge_indices = None

        # Find clusters with maximum single linkage distance and merge them
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = single_linkage(clusters[i], clusters[j])
                if distance > max_distance:
                    max_distance = distance
                    merge_indices = (i, j)

        merged_cluster = clusters[merge_indices[0]] + clusters[merge_indices[1]]
        merged_labels = cluster_labels[merge_indices[0]] + cluster_labels[merge_indices[1]]

        clusters[merge_indices[0]] = merged_cluster
        cluster_labels[merge_indices[0]] = merged_labels

        del clusters[merge_indices[1]]
        del cluster_labels[merge_indices[1]]
        iteration += 1
        print(iteration)

    return clusters, cluster_labels

# Function for k-means clustering
def kmeans_clustering(data, k, max_iterations=20):
    np.random.seed(42)
    indices = np.random.choice(data.shape[0], k, replace=False)
    cluster_means = data[indices]
    cluster_assignments = np.zeros(data.shape[0], dtype=int)

    for iteration in range(max_iterations):
        new_assignments = []

        # Assign points to clusters based on cosine similarity with cluster means
        for point in data:
            similarities = [cosine_similarity(point, mean) for mean in cluster_means]
            closest_cluster = np.argmax(similarities)
            new_assignments.append(closest_cluster)

        new_assignments = np.array(new_assignments)

        # Check for convergence
        if np.array_equal(new_assignments, cluster_assignments):
            break

        cluster_assignments = new_assignments

        # Update cluster means based on assigned points
        for i in range(k):
            cluster_points = data[cluster_assignments == i]
            if len(cluster_points) > 0:
                cluster_means[i] = np.mean(cluster_points, axis=0)

    # Generate cluster labels for each point
    cluster_labels = [[] for _ in range(k)]
    for idx, label in enumerate(cluster_assignments):
        cluster_labels[label].append(idx + 1)  # Using 1-based index for labels

    return cluster_assignments, cluster_labels

# Function to calculate cosine similarity between two points
def cosine_similarity(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)

    dot_product = np.dot(point1, point2)
    norm_point1 = np.linalg.norm(point1)
    norm_point2 = np.linalg.norm(point2)

    if norm_point1 == 0 or norm_point2 == 0:
        return 1.0

    cosine_similarity = dot_product / (norm_point1 * norm_point2)
    return 1 - cosine_similarity

# Function to calculate silhouette coefficient for clustering evaluation
def silhouette_coefficient(data, cluster_assignments):
    silhouette_scores = []
    for i, point in enumerate(data):
        cluster = cluster_assignments[i]
        a = np.mean([cosine_similarity(point, data[j]) for j in range(len(data)) if cluster_assignments[j] == cluster and j != i])

        other_clusters = set(cluster_assignments) - {cluster}
        b_values = []
        for other_cluster in other_clusters:
            b_values.append(np.mean([cosine_similarity(point, data[j]) for j in range(len(data)) if cluster_assignments[j] == other_cluster]))

        b = np.min(b_values) if len(b_values) > 0 else 0
        silhouette = (b - a) / max(a, b)
        silhouette_scores.append(silhouette)

    mean_silhouette = np.mean(silhouette_scores)
    return mean_silhouette

# Function to calculate Jaccard similarity between cluster labels
def calculate_jaccard_similarity(cluster_labels1, cluster_labels2):
    jaccard_scores = []
    for labels1 in cluster_labels1:
        for labels2 in cluster_labels2:
            jaccard_score = len(set(labels1) & set(labels2)) / len(set(labels1) | set(labels2))
            jaccard_scores.append(jaccard_score)

    return max(jaccard_scores)

# Main function to run clustering algorithms and evaluate results
def main():
    start_time = time.time()
    data, _ = read_data('preferences.csv')

    k_values = [3, 4, 5, 6]
    max_silhouette = -1
    optimal_k = None

    # Perform k-means clustering for different values of k and select optimal k based on silhouette coefficient
    for k in k_values:
        cluster_assignments, cluster_labels = kmeans_clustering(data, k)
        silhouette = silhouette_coefficient(data, cluster_assignments)
        print(f'Silhouette coefficient for k={k}: {silhouette}')
        if silhouette > max_silhouette:
            max_silhouette = silhouette
            optimal_k = k

    print(f'Optimal value of k: {optimal_k} (highest Silhouette Coefficient)')

    # Run k-means clustering with optimal k and print final clusters
    cluster_assignments, cluster_labels = kmeans_clustering(data, optimal_k)
    print("\nFinal clusters (K-Means):")
    for i, labels in enumerate(cluster_labels):
        print(f"Cluster {i+1}: {labels}")
        print(f"Number of points: {len(labels)}\n")

    # Save k-means clusters to a text file
    with open('kmeans.txt', 'w') as file:
        for i, labels in enumerate(cluster_labels):
            file.write(f"Cluster {i+1}: {labels}\n")
            file.write(f"Number of points: {len(labels)}\n\n")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Time taken for K-means: {elapsed_time} seconds")

    # Run hierarchical clustering with optimal k and print final clusters
    start_time = time.time()
    final_clusters, final_labels = hierarchical_clustering(data, optimal_k)

    print("\nFinal clusters (Hierarchical):")
    for i, labels in enumerate(final_labels):
        print(f"Cluster {i}: {labels}")
        print(f"Number of points: {len(labels)}\n")

    # Save hierarchical clusters to a text file
    with open('agglomerative.txt', 'w') as file:
        for i, labels in enumerate(final_labels):
            file.write(f"Cluster {i}: {labels}\n")
            file.write(f"Number of points: {len(labels)}\n\n")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Time taken for Agglomerative: {elapsed_time} seconds")

    # Calculate Jaccard similarity between k-means and hierarchical clusters
    jaccard_similarity_scores = []
    for labels in cluster_labels:
        max_similarity = -1
        for hierarchical_labels in final_labels:
            similarity = calculate_jaccard_similarity([labels], [hierarchical_labels])
            if similarity > max_similarity:
                max_similarity = similarity
        jaccard_similarity_scores.append(max_similarity)

    print(f"Jaccard Similarity : {jaccard_similarity_scores}")


if __name__ == "__main__":
    main()
