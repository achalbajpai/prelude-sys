# Medical Records Page Clustering - User Guide

This guide provides detailed instructions for using the Medical Records Page Clustering solution, which automates the process of grouping pages from unstructured medical record PDFs.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Advanced Usage](#advanced-usage)
4. [Understanding Outputs](#understanding-outputs)
5. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

-  Python 3.8 or higher
-  Tesseract OCR for text extraction from image-based PDFs
-  Poppler (optional but recommended for enhanced PDF processing)

### Step-by-Step Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/medical-records-clustering.git
   cd medical-records-clustering
   ```

2. **Run the setup script:**

   ```bash
   # Make setup script executable
   chmod +x setup.sh

   # Run setup script
   ./setup.sh
   ```

   The setup script will:

   -  Install required Python packages
   -  Install Tesseract OCR (if on macOS or Linux)
   -  Download the necessary spaCy language model

3. **Manual dependencies (if setup script fails):**
   -  Install Python dependencies: `pip install -r requirements.txt`
   -  Install Tesseract OCR:
      -  macOS: `brew install tesseract`
      -  Ubuntu/Debian: `apt-get install tesseract-ocr`
      -  Windows: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   -  Install spaCy language model: `python -m spacy download en_core_web_sm`
   -  Install Poppler (optional):
      -  macOS: `brew install poppler`
      -  Ubuntu/Debian: `apt-get install poppler-utils`
      -  Windows: Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases/)

## Basic Usage

### Quick Demo

To run a quick demonstration with the sample document:

```bash
python demo.py
```

This will:

1. Process the sample PDF document
2. Group pages into clusters
3. Create visualizations of the clustering
4. Output results to `demo_output/` directory

### Clustering a PDF Document

To cluster pages in your own PDF document:

```bash
python main.py --pdf "path/to/your/document.pdf" --output "results.csv"
```

## Advanced Usage

### Using Sample Data for Validation and Optimization

If you have sample data that indicates the correct grouping of pages, you can use it to validate and optimize the clustering:

```bash
python main.py --pdf "path/to/your/document.pdf" --sample "path/to/sample_data.csv" --output "results.csv" --optimize
```

The `--optimize` flag will determine the optimal similarity threshold for clustering based on your sample data.

### Command Line Arguments

| Argument     | Description                                   | Default Value                   |
| ------------ | --------------------------------------------- | ------------------------------- |
| `--pdf`      | Path to the PDF file to analyze               | "Sample Document.pdf"           |
| `--sample`   | Path to sample data CSV for validation        | "Sample Data.csv"               |
| `--output`   | Path to save the output CSV file              | "Medical_Records_Clustered.csv" |
| `--optimize` | Optimize clustering threshold based on sample | False (flag, no value)          |
| `--viz`      | Prefix for visualization files                | "clustering_results"            |

### Sample Data Format

If you have sample data for validation, it should be in CSV format with the following columns:

-  `pagenumber`: Page number in the PDF
-  `parentkey`: ID of the parent record (0 if this is the first page of a record)
-  `referencekey`: Unique ID for this page
-  `header`: Document header
-  `category`: Document category (e.g., 24 for Lab Report, 16 for Progress Note)
-  `dos`: Date of service

Example:

```csv
pagenumber,category,isreviewable,dos,provider,referencekey,parentkey,lockstatus,header,facilitygroup,reviewerid,qcreviewerid,isduplicate
1,24,TRUE,4/17/2019,Doctor Name,120991,0,L,Laboratory Rept,,287,322,FALSE
2,24,TRUE,4/17/2019,Doctor Name,120992,120991,L,Laboratory Rept,,287,322,FALSE
```

## Understanding Outputs

### CSV Output

The output CSV file contains:

-  `pagenumber`: Page number in the PDF
-  `parentkey`: ID of the parent record (0 if this is the first page of a record)
-  `referencekey`: Unique ID for this page
-  `header`: Detected document header
-  `category`: Detected document category
-  `dos`: Detected date of service
-  `provider`: Detected provider information
-  `facility`: Detected facility information

### Visualizations

The solution generates three visualization files:

1. `*_clusters.png`: Visual representation of the page clusters
2. `*_comparison.png`: Comparison of predicted clusters vs. sample data (if provided)
3. `*_threshold.png`: Performance of different similarity thresholds (if optimization is enabled)

## Troubleshooting

### Common Issues

1. **Text extraction issues:**

   -  Ensure Tesseract OCR is properly installed
   -  Check that the PDF isn't password-protected
   -  Try using a higher quality PDF if available

2. **Dependencies not found:**

   -  Run `pip install -r requirements.txt` again
   -  Check if external dependencies (Tesseract, Poppler) are installed correctly

3. **Poor clustering results:**
   -  Try providing sample data and use the `--optimize` flag
   -  Adjust the threshold manually in `page_clustering.py` if needed
   -  Consider preprocessing the PDF if it has poor quality or formatting

### Getting Help

If you encounter issues not covered in this guide, please:

1. Check the project's GitHub issues
2. Submit a new issue with a detailed description of the problem
3. Include error messages and examples if possible
