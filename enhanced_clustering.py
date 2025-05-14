import os
import numpy as np
import pandas as pd
from page_clustering import MedicalRecordPageClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import joblib


class EnhancedMedicalRecordClustering:
    def __init__(self, pdf_path, sample_data_path=None):
        self.pdf_path = pdf_path
        self.sample_data_path = sample_data_path
        self.base_clustering = MedicalRecordPageClustering(pdf_path)
        self.sample_data = None
        self.prediction_metrics = {}

        if sample_data_path:
            self.sample_data = pd.read_csv(sample_data_path)

    def optimize_threshold(self, thresholds=None):
        """Find optimal similarity threshold for clustering"""
        if thresholds is None:
            thresholds = np.arange(0.3, 0.9, 0.05)

        # Extract text and features if not already done
        if not self.base_clustering.page_texts:
            self.base_clustering.extract_text_from_pdf()
            self.base_clustering.extract_page_features()

        # Compute similarity matrix once
        similarity_matrix, page_nums = self.base_clustering.compute_page_similarities()

        # If we have sample data, use it to validate
        if self.sample_data is not None:
            # Create ground truth clusters from sample data
            ground_truth = {}
            for _, row in self.sample_data.iterrows():
                page_num = row["pagenumber"]
                parent_key = (
                    row["parentkey"] if row["parentkey"] != 0 else row["referencekey"]
                )

                if parent_key not in ground_truth:
                    ground_truth[parent_key] = []
                ground_truth[parent_key].append(page_num)

            # Test different thresholds
            best_score = -1
            best_threshold = thresholds[0]
            scores = []

            print("Optimizing clustering threshold...")
            for threshold in thresholds:
                # Cluster with current threshold
                clusters = self.cluster_with_threshold(
                    similarity_matrix, page_nums, threshold
                )

                # Convert to format for scoring
                pred_labels = np.zeros(max(page_nums) + 1)
                for cluster_id, pages in clusters.items():
                    for page in pages:
                        pred_labels[page] = cluster_id

                # Convert ground truth to same format
                true_labels = np.zeros(max(page_nums) + 1)
                for cluster_id, pages in ground_truth.items():
                    for page in pages:
                        true_labels[page] = cluster_id

                # Calculate ARI score (adjusted for chance)
                valid_indices = np.where((true_labels > 0) & (pred_labels > 0))[0]
                if len(valid_indices) > 1:
                    ari_score = adjusted_rand_score(
                        true_labels[valid_indices], pred_labels[valid_indices]
                    )
                    scores.append((threshold, ari_score))

                    if ari_score > best_score:
                        best_score = ari_score
                        best_threshold = threshold

            # Plot threshold vs score
            if scores:
                thresholds, scores = zip(*scores)
                plt.figure(figsize=(10, 6))
                plt.plot(thresholds, scores, marker="o")
                plt.axvline(
                    x=best_threshold,
                    color="r",
                    linestyle="--",
                    label=f"Best threshold: {best_threshold:.2f}",
                )
                plt.grid(True, alpha=0.3)
                plt.xlabel("Similarity Threshold")
                plt.ylabel("Adjusted Rand Index Score")
                plt.title("Clustering Performance vs. Threshold")
                plt.legend()
                plt.savefig("threshold_optimization.png")

                print(
                    f"Best threshold: {best_threshold:.2f} with ARI score: {best_score:.4f}"
                )
                return best_threshold

        # Default to middle threshold if no sample data
        return 0.6

    def cluster_with_threshold(self, similarity_matrix, page_nums, threshold):
        """Cluster pages with given threshold"""
        n_pages = len(page_nums)

        # Initialize clusters
        clusters = {}
        cluster_id = 1
        assigned_pages = set()

        # Group pages based on similarity
        for i in range(n_pages):
            if page_nums[i] in assigned_pages:
                continue

            current_cluster = [page_nums[i]]
            assigned_pages.add(page_nums[i])

            # current_matrix_idx is the index in similarity_matrix and page_nums for the last page added
            current_matrix_idx = i
            while current_matrix_idx < n_pages - 1:
                page_current_num = page_nums[current_matrix_idx]
                page_next_num = page_nums[current_matrix_idx + 1]

                # Access page_features from base_clustering
                doc_type_current = self.base_clustering.page_features[page_current_num][
                    "doc_type"
                ]
                doc_type_next = self.base_clustering.page_features[page_next_num][
                    "doc_type"
                ]

                force_break = False
                if (
                    doc_type_current != "unknown"
                    and doc_type_next != "unknown"
                    and doc_type_current != doc_type_next
                ):
                    force_break = True

                if (
                    not force_break
                    and similarity_matrix[current_matrix_idx, current_matrix_idx + 1]
                    >= threshold
                ):
                    current_cluster.append(page_next_num)
                    assigned_pages.add(page_next_num)
                    current_matrix_idx += 1
                else:
                    break

            clusters[cluster_id] = current_cluster
            cluster_id += 1

        return clusters

    def validate_against_sample(self, prediction_df):
        """Validate clustering results against sample data"""
        if self.sample_data is None:
            print("No sample data available for validation")
            return {}

        print("Validating against sample data...")

        # Create mapping of page to cluster ID
        pred_clusters = {}
        for _, row in prediction_df.iterrows():
            page_num = row["pagenumber"]
            parent_key = (
                row["parentkey"] if row["parentkey"] != 0 else row["referencekey"]
            )
            pred_clusters[page_num] = parent_key

        # Create mapping from sample data
        true_clusters = {}
        for _, row in self.sample_data.iterrows():
            page_num = row["pagenumber"]
            parent_key = (
                row["parentkey"] if row["parentkey"] != 0 else row["referencekey"]
            )
            true_clusters[page_num] = parent_key

        # Get common pages
        common_pages = set(pred_clusters.keys()).intersection(set(true_clusters.keys()))

        if not common_pages:
            print("No common pages found between prediction and sample data")
            return {}

        # Extract cluster assignments for common pages
        y_true = [true_clusters[p] for p in common_pages]
        y_pred = [pred_clusters[p] for p in common_pages]

        # Convert to numeric labels for scoring
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_true_encoded = le.fit_transform(y_true)

        le2 = LabelEncoder()
        y_pred_encoded = le2.fit_transform(y_pred)

        # Calculate metrics
        ari = adjusted_rand_score(y_true_encoded, y_pred_encoded)
        nmi = normalized_mutual_info_score(y_true_encoded, y_pred_encoded)

        # Count correctly grouped pages
        cluster_map = {}
        for p in common_pages:
            true_cluster = true_clusters[p]
            pred_cluster = pred_clusters[p]

            if true_cluster not in cluster_map:
                cluster_map[true_cluster] = {}

            if pred_cluster not in cluster_map[true_cluster]:
                cluster_map[true_cluster][pred_cluster] = 0

            cluster_map[true_cluster][pred_cluster] += 1

        # Find best mapping between true and predicted clusters
        correct_pages = 0
        total_pages = len(common_pages)

        for true_cluster, pred_counts in cluster_map.items():
            best_pred = max(pred_counts.items(), key=lambda x: x[1])
            correct_pages += best_pred[1]

        accuracy = correct_pages / total_pages if total_pages > 0 else 0

        metrics = {
            "adjusted_rand_index": ari,
            "normalized_mutual_info": nmi,
            "accuracy": accuracy,
            "correctly_grouped_pages": correct_pages,
            "total_pages": total_pages,
        }

        self.prediction_metrics = metrics

        print(f"Validation Results:")
        print(f"Adjusted Rand Index: {ari:.4f}")
        print(f"Normalized Mutual Information: {nmi:.4f}")
        print(
            f"Grouping Accuracy: {accuracy:.2%} ({correct_pages}/{total_pages} pages)"
        )

        return metrics

    def visualize_comparison(self, prediction_df, output_path="cluster_comparison.png"):
        """Visualize comparison between predicted and true clusters"""
        if self.sample_data is None:
            print("No sample data available for comparison visualization")
            return

        print("Generating comparison visualization...")

        # Get max page number
        max_page = max(
            max(self.sample_data["pagenumber"]), max(prediction_df["pagenumber"])
        )

        # Create arrays for true and predicted clusters
        true_clusters = np.zeros(max_page + 1)
        pred_clusters = np.zeros(max_page + 1)

        # Fill true clusters
        for _, row in self.sample_data.iterrows():
            page = row["pagenumber"]
            parent = row["parentkey"] if row["parentkey"] != 0 else 0
            if parent == 0:  # Start of a cluster
                true_clusters[page] = row["referencekey"]
            else:
                true_clusters[page] = parent

        # Fill predicted clusters
        for _, row in prediction_df.iterrows():
            page = row["pagenumber"]
            parent = row["parentkey"] if row["parentkey"] != 0 else 0
            if parent == 0:  # Start of a cluster
                pred_clusters[page] = row["referencekey"]
            else:
                pred_clusters[page] = parent

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot true clusters
        unique_true = np.unique(true_clusters[true_clusters > 0])
        colors_true = plt.cm.tab20(np.linspace(0, 1, len(unique_true)))
        color_map_true = {
            cluster: colors_true[i] for i, cluster in enumerate(unique_true)
        }

        x_true = np.arange(1, max_page + 1)
        y_true = np.ones_like(x_true)

        for i, page in enumerate(x_true):
            if true_clusters[page] > 0:
                cluster = true_clusters[page]
                ax1.scatter(page, 1, color=color_map_true[cluster], s=100)

                # Draw line connecting pages in same cluster
                if i > 0 and true_clusters[page - 1] == cluster:
                    ax1.plot(
                        [page - 1, page],
                        [1, 1],
                        color=color_map_true[cluster],
                        alpha=0.5,
                    )

        # Plot predicted clusters
        unique_pred = np.unique(pred_clusters[pred_clusters > 0])
        colors_pred = plt.cm.tab20(np.linspace(0, 1, len(unique_pred)))
        color_map_pred = {
            cluster: colors_pred[i] for i, cluster in enumerate(unique_pred)
        }

        x_pred = np.arange(1, max_page + 1)
        y_pred = np.ones_like(x_pred)

        for i, page in enumerate(x_pred):
            if pred_clusters[page] > 0:
                cluster = pred_clusters[page]
                ax2.scatter(page, 1, color=color_map_pred[cluster], s=100)

                # Draw line connecting pages in same cluster
                if i > 0 and pred_clusters[page - 1] == cluster:
                    ax2.plot(
                        [page - 1, page],
                        [1, 1],
                        color=color_map_pred[cluster],
                        alpha=0.5,
                    )

        # Set labels and titles
        ax1.set_title("True Clusters (Sample Data)")
        ax1.set_yticks([])

        ax2.set_title("Predicted Clusters")
        ax2.set_xlabel("Page Number")
        ax2.set_yticks([])

        ax1.grid(True, axis="x", alpha=0.3)
        ax2.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Comparison visualization saved to {output_path}")

    def process(self, output_csv_path, optimize=True):
        """Run the complete enhanced clustering process"""
        # Extract text and features
        self.base_clustering.extract_text_from_pdf()
        self.base_clustering.extract_page_features()

        # Optimize threshold if requested
        if optimize and self.sample_data is not None:
            optimal_threshold = self.optimize_threshold()
            self.base_clustering.cluster_pages(threshold=optimal_threshold)
        else:
            self.base_clustering.cluster_pages()

        # Generate output CSV
        result_df = self.base_clustering.generate_output_csv(output_csv_path)

        # Validate against sample data if available
        if self.sample_data is not None:
            self.validate_against_sample(result_df)
            self.visualize_comparison(result_df)

        # Visualize clusters
        self.base_clustering.visualize_clusters("Page_Clusters_Visualization.png")

        return result_df, self.prediction_metrics


# Example usage
if __name__ == "__main__":
    pdf_path = "Sample Document.pdf"
    sample_data_path = "Sample Data.csv"
    output_csv = "Enhanced_Clustered_Medical_Records.csv"

    enhanced_clustering = EnhancedMedicalRecordClustering(pdf_path, sample_data_path)
    result_df, metrics = enhanced_clustering.process(output_csv, optimize=True)

    print("Enhanced page clustering complete!")
    print(
        f"Found {len(enhanced_clustering.base_clustering.page_clusters)} distinct medical record groups"
    )

    if metrics:
        print(f"Final validation metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
