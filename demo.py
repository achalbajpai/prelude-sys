import os
import argparse
from enhanced_clustering import EnhancedMedicalRecordClustering
import matplotlib.pyplot as plt
import pandas as pd


def run_demo(pdf_path="Sample Document.pdf", sample_path="Sample Data.csv"):
    """Run a quick demo of the medical records page clustering solution"""

    print("\n" + "=" * 80)
    print("MEDICAL RECORDS PAGE CLUSTERING - QUICK DEMO")
    print("=" * 80)

    # Check if files exist
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return

    # Check if sample data exists
    use_sample = os.path.exists(sample_path)
    if not use_sample:
        print(
            f"Warning: Sample data file not found at {sample_path}. Will proceed without validation."
        )

    # Create output directory
    if not os.path.exists("demo_output"):
        os.makedirs("demo_output")

    # Output file paths
    output_csv = "demo_output/clustered_results.csv"

    # Initialize clustering
    sample_path_to_use = sample_path if use_sample else None

    print(f"Processing PDF: {pdf_path}")
    if use_sample:
        print(f"Using sample data for validation: {sample_path}")
    print(f"Results will be saved to: {output_csv}")
    print("\nStarting page clustering...")

    # Run enhanced clustering
    clustering = EnhancedMedicalRecordClustering(
        pdf_path=pdf_path, sample_data_path=sample_path_to_use
    )

    # Process the document with optimization if sample data is available
    result_df, metrics = clustering.process(
        output_csv_path=output_csv, optimize=use_sample
    )

    # Print summary
    print("\n" + "=" * 80)
    print("CLUSTERING RESULTS")
    print("=" * 80)

    clusters = clustering.base_clustering.page_clusters
    print(f"Total pages processed: {sum(len(pages) for pages in clusters.values())}")
    print(f"Number of clusters identified: {len(clusters)}")

    # Print each cluster details
    for cluster_id, pages in clusters.items():
        metadata = clustering.base_clustering.extract_record_metadata(pages)
        header = metadata["header"] or "Unknown"
        dos = metadata["dos"] or "Unknown"
        category = metadata["category"] or "Unknown"

        print(f"\nCluster {cluster_id}:")
        print(f"  Pages: {pages}")
        print(f"  Range: {pages[0]}-{pages[-1]}")
        print(f"  Header: {header}")
        print(f"  Date of Service: {dos}")
        print(f"  Category: {category}")

    # Rename visualization files to the demo_output directory
    viz_files = {
        "Page_Clusters_Visualization.png": "demo_output/clusters_visualization.png",
        "cluster_comparison.png": "demo_output/clusters_comparison.png",
        "threshold_optimization.png": "demo_output/threshold_optimization.png",
    }

    for old_name, new_name in viz_files.items():
        if os.path.exists(old_name):
            os.rename(old_name, new_name)
            print(f"\nVisualization saved to {new_name}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print(f"Results saved to {output_csv}")
    print("Visualizations saved to the demo_output directory")
    print("=" * 80)

    return result_df, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a quick demo of the Medical Records Page Clustering solution",
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
        help="Path to the sample data CSV file for validation",
    )

    args = parser.parse_args()

    try:
        result_df, metrics = run_demo(pdf_path=args.pdf, sample_path=args.sample)
    except Exception as e:
        print(f"Error during demo: {str(e)}")
        import traceback

        traceback.print_exc()
