#!/bin/bash

# Exit on error
set -e

echo "======================================================"
echo "Medical Records Page Clustering - Setup"
echo "======================================================"

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed. Please install Python and pip first."
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check OS and install Tesseract OCR if needed
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Detected macOS. Installing Tesseract OCR via Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Please install Homebrew first or install Tesseract manually."
        echo "Visit: https://github.com/tesseract-ocr/tesseract"
    else
        if ! command -v tesseract &> /dev/null; then
            brew install tesseract
        else
            echo "Tesseract OCR already installed."
        fi
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "Detected Linux. Installing Tesseract OCR..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr
    elif command -v yum &> /dev/null; then
        sudo yum install -y tesseract
    else
        echo "Could not detect package manager. Please install Tesseract OCR manually."
        echo "Visit: https://github.com/tesseract-ocr/tesseract"
    fi
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    echo "Detected Windows. Please install Tesseract OCR manually from:"
    echo "https://github.com/UB-Mannheim/tesseract/wiki"
    echo "After installation, make sure to add Tesseract to your PATH."
fi

# Install spaCy language model
echo "Installing spaCy language model..."
python -m spacy download en_core_web_sm

echo "======================================================"
echo "Setup complete! You can now run the page clustering tool."
echo "======================================================"
echo "Usage: python main.py --pdf \"Sample Document.pdf\" --optimize"
echo "======================================================" 