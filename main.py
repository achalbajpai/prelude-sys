import os
import argparse
from enhanced_clustering import EnhancedMedicalRecordClustering
import pandas as pd


def main():
    """Main function to run the medical record page clustering"""

    # Set up command line arguments
    parser = argparse.ArgumentParser(
        description="Medical Record Page Clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--pdf",
        type=str,
        default="Sample Document.pdf",
        help="Path to the PDF file to analyze",
    )

    parser.add_argument(
        "--sample",
        type=str,
        default="Sample Data.csv",
        help="Path to the sample data CSV file (optional)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="Medical_Records_Clustered.csv",
        help="Path to save the output CSV file",
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize clustering threshold based on sample data",
    )

    parser.add_argument(
        "--viz",
        type=str,
        default="clustering_results",
        help="Prefix for visualization files",
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.pdf):
        print(f"Error: PDF file not found at {args.pdf}")
        return

    use_sample = os.path.exists(args.sample)
    if not use_sample and args.optimize:
        print(
            f"Warning: Sample data file not found at {args.sample}, cannot optimize threshold"
        )

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize clustering
    print("\n" + "=" * 80)
    print(f"MEDICAL RECORD PAGE CLUSTERING")
    print("=" * 80)
    print(f"PDF file: {args.pdf}")
    if use_sample:
        print(f"Sample data: {args.sample}")
    print(f"Output file: {args.output}")
    print(f"Optimize threshold: {args.optimize}")
    print("=" * 80 + "\n")

    # Run enhanced clustering
    sample_path = args.sample if use_sample else None

    clustering = EnhancedMedicalRecordClustering(
        pdf_path=args.pdf, sample_data_path=sample_path
    )

    # Process the document
    result_df, metrics = clustering.process(
        output_csv_path=args.output, optimize=args.optimize
    )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    clusters = clustering.base_clustering.page_clusters
    print(f"Total pages processed: {sum(len(pages) for pages in clusters.values())}")
    print(f"Number of clusters: {len(clusters)}")

    # Print each cluster
    for cluster_id, pages in clusters.items():
        metadata = clustering.base_clustering.extract_record_metadata(pages)
        header = metadata["header"] or "Unknown"
        dos = metadata["dos"] or "Unknown"
        category = metadata["category"] or "Unknown"

        print(
            f"  Cluster {cluster_id}: {len(pages)} pages [{pages[0]}-{pages[-1]}] - {header} (DOS: {dos}, Category: {category})"
        )

    # Print validation metrics if available
    if metrics:
        print("\nValidation Metrics:")
        print(
            f"  Grouping Accuracy: {metrics['accuracy']:.2%} ({metrics['correctly_grouped_pages']}/{metrics['total_pages']} pages)"
        )
        print(f"  Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f}")
        print(
            f"  Normalized Mutual Information: {metrics['normalized_mutual_info']:.4f}"
        )

    print("=" * 80)
    print(f"Results saved to {args.output}")

    # Rename visualization files with prefix
    viz_files = {
        "Page_Clusters_Visualization.png": f"{args.viz}_clusters.png",
        "cluster_comparison.png": f"{args.viz}_comparison.png",
        "threshold_optimization.png": f"{args.viz}_threshold.png",
    }

    for old_name, new_name in viz_files.items():
        if os.path.exists(old_name):
            os.rename(old_name, new_name)
            print(f"Visualization saved to {new_name}")

    print("=" * 80)
    return result_df


if __name__ == "__main__":
    try:
        result_df = main()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
