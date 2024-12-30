#!/bin/bash

# Check if pdftotext command exists
if ! command -v pdftotext &> /dev/null; then
    echo "Error: pdftotext command not found. Make sure poppler-utils is installed."
    exit 1
fi

# Define the directory containing PDF files
pdf_dir="/path/to/pdf"

txt_dir="/path/to/txt"
# Check if the directory exists
if [ ! -d "$pdf_dir" ]; then
    echo "Error: Directory $pdf_dir not found."
    exit 1
fi

# Loop through each PDF file in the directory
for pdf_file in "$pdf_dir"/*.pdf; do
    # Extract the filename without extension
    filename=$(basename -- "$pdf_file")
    filename_no_ext="${filename%.*}"

    # Convert PDF to text using pdftotext command
    pdftotext "$pdf_file" "$txt_dir/$filename_no_ext.txt"

    # Check if pdftotext command was successful
    if [ $? -eq 0 ]; then
        echo "Converted $pdf_file to text: $txt_dir/$filename_no_ext.txt"
    else
        echo "Error: Failed to convert $pdf_file to text."
    fi
done

