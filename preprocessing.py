import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import os

def preprocess_text(text):
    """
    Cleans the text by removing unknown characters, punctuation, and unnecessary spaces.

    Args:
        text (str): Raw text input.

    Returns:
        str: Cleaned text.
    """
    # Remove non-UTF-8 or unknown characters
    text = text.encode("utf-8", "ignore").decode("utf-8")
    
    # Remove special characters and punctuation (but keep periods for sentence splitting)
    text = re.sub(r"[^\w\s\.\!\?]", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def intelligent_cleaning(text, num_clusters=3):
    """
    Cleans a text file intelligently by identifying the main content.

    Args:
        text (str): Full text of the book.
        num_clusters (int): Number of clusters to form for segmentation.

    Returns:
        str: Cleaned text containing only the main content.
    """
    # Step 1: Preprocess the raw text
    cleaned_text = preprocess_text(text)

    # Step 2: Split into sentences
    sentences = re.split(r"(?<=[\.\!\?])\s", cleaned_text)  # Split on sentence-ending punctuation
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    
    # If the number of sentences is less than or equal to the number of clusters, return the cleaned text
    if len(sentences) <= num_clusters:
        return cleaned_text

    # Step 3: Generate embeddings for sentences
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight transformer for sentence embeddings
    embeddings = model.encode(sentences)
    
    # Step 4: Cluster sentences
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    # Step 5: Find the cluster with the highest density (assumed main content)
    cluster_sizes = np.bincount(labels)
    main_cluster = np.argmax(cluster_sizes)
    
    # Step 6: Extract sentences from the main cluster
    main_content = [sentences[i] for i in range(len(sentences)) if labels[i] == main_cluster]
    
    # Step 7: Combine sentences into a single text block
    return " ".join(main_content)

# Directory containing .txt files
directory = 'txt'

# Log file to keep track of processed files
log_file = 'processed_files.log'

# Load processed files log
processed_files = {}
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        for line in f:
            filename, char_count = line.strip().split(': ')
            processed_files[filename] = int(char_count)

# Process all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        
        # Check if the file has already been processed
        if filename in processed_files:
            print(f"Skipping {filename}: already processed with {processed_files[filename]} characters retained.")
            continue
        
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean the text
        cleaned_text = intelligent_cleaning(text)
        
        # Save the cleaned text
        cleaned_file_path = os.path.join(directory+'/cleaned/', f"cleaned_{filename}")
        with open(cleaned_file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        # Log the processed file
        char_count = len(cleaned_text)
        with open(log_file, 'a') as f:
            f.write(f"{filename}: {char_count}\n")
        
        print(f"Processed {filename}: {char_count} characters retained.")
