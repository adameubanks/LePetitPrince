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

# Process all files in the directory
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Preprocess and intelligently clean the text
    processed_text = intelligent_cleaning(raw_text)
    print(f"Processed {filename}: {len(processed_text)} characters retained.")
