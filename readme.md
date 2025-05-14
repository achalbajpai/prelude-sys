# Medical Records Page Clustering

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning solution for grouping pages from unstructured medical record PDF documents. The system analyzes text content and structure to identify which pages belong together, even when they are interspersed with other information.

## 🔍 Problem Overview

Medical history documents often contain thousands of pages with different types of medical records (lab reports, progress notes, etc.) that need to be properly grouped together. The main challenges include:

-  Identifying and grouping pages that belong to the same record
-  Carrying forward metadata like date of service (DOS), header, provider, and facility information
-  Handling missing information across multiple pages
-  Managing various record types and lengths

## ✨ Features

-  **PDF Text Extraction**: Extracts text using both direct extraction and OCR fallback for image-based PDFs
-  **Feature Engineering**: Analyzes page content, structure, headers, and entities to determine page similarity
-  **Clustering Algorithm**: Groups related pages based on content and structural similarity
-  **Metadata Extraction**: Identifies dates of service, providers, facility information, and document types
-  **Validation**: Compares results against sample data to validate clustering accuracy
-  **Visualization**: Creates visual representations of page clusters and comparisons

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/achalbajpai/prelude-sys.git
cd prelude-sys
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### External Dependencies

For OCR functionality, you need to install Tesseract OCR:

-  On macOS: `brew install tesseract`
-  On Ubuntu/Debian: `apt-get install tesseract-ocr`
-  On Windows: [Download and install Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

For enhanced PDF processing (optional):

-  On macOS: `brew install poppler`
-  On Ubuntu/Debian: `apt-get install poppler-utils`
-  On Windows: Download from [here](https://github.com/oschwartz10612/poppler-windows/releases/)

## 📋 Usage

### Basic Usage

```bash
python main.py --pdf "your_document.pdf" --output "results.csv"
```

### With Sample Data for Validation

```bash
python main.py --pdf "your_document.pdf" --sample "your_sample_data.csv" --output "results.csv" --optimize
```

### Command Line Arguments

-  `--pdf`: Path to the PDF file to analyze
-  `--sample`: Path to sample data CSV for validation (optional)
-  `--output`: Path to save the output CSV file
-  `--optimize`: Flag to optimize clustering threshold based on sample data
-  `--viz`: Prefix for visualization files

### Quick Demo

For a quick demonstration with visualization outputs:

```bash
python demo.py --pdf "your_document.pdf"
```

## 📊 Output

The solution generates:

1. **CSV output file** with page grouping information and metadata
2. **Visualization files** showing the clustering results and validation

## 🧠 How It Works

1. **Text Extraction**: Uses PyPDF2 with OCR fallback to extract text from each page
2. **Feature Extraction**: Analyzes pages to identify document type, structure, and metadata
3. **Similarity Computation**: Calculates similarity between pages using TF-IDF and cosine similarity
4. **Clustering**: Groups pages based on similarity scores
5. **Metadata Assignment**: Extracts and assigns metadata to each cluster
6. **Validation & Visualization**: Validates against sample data and generates visualizations

## 📝 Project Files

-  `page_clustering.py`: Base clustering implementation
-  `enhanced_clustering.py`: Extended clustering with validation and optimization
-  `main.py`: Command-line interface for running the solution
-  `demo.py`: Quick demonstration script with visualizations
-  `setup.sh`: Setup script for installing dependencies
-  `requirements.txt`: List of Python dependencies
-  `USER_GUIDE.md`: Detailed user guide with instructions
-  `verification_report.md`: Verification of requirements implementation
-  `FUTURE_ENHANCEMENTS.md`: Roadmap for future improvements

## ⚙️ Performance

The solution is designed to handle large medical record PDFs with thousands of pages. For optimal performance:

-  Use direct text extraction where possible (avoid OCR)
-  Run on a machine with sufficient RAM for large documents
-  Consider batch processing for extremely large files

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or support, please open an issue on GitHub.
