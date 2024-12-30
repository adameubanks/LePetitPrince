import os
import glob
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import sentencepiece as spm

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def create_sequences(text, sequence_length, spm_model):
    token_list = spm_model.EncodeAsIds(text)
    input_sequences = []
    for i in range(sequence_length, len(token_list)):
        seq = token_list[i-sequence_length:i]
        input_sequences.append(seq)
    input_sequences = np.array(input_sequences)
    return input_sequences, len(spm_model)

def create_autoencoder(sequence_length, total_words):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(sequence_length, total_words)))
    model.add(tf.keras.layers.LSTM(100, return_sequences=False))
    model.add(tf.keras.layers.RepeatVector(sequence_length))
    model.add(tf.keras.layers.LSTM(100, return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(total_words, activation='softmax')))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

# Directory containing .txt files
directory = os.path.join('txt', 'cleaned')
input_files = glob.glob(os.path.join(directory, '*.txt'))

# File to save results
results_file = 'results.txt'
trained_files_log = 'trained_files.log'

# Initialize SentencePiece model for tokenization
spm.SentencePieceTrainer.train(input=input_files, model_prefix='spm', vocab_size=8000, character_coverage=1.0)
sp = spm.SentencePieceProcessor(model_file='spm.model')

# Load processed files log
processed_files = {}
if os.path.exists(trained_files_log):
    with open(trained_files_log, 'r') as f:
        for line in f:
            filename, loss = line.strip().split(': ')
            processed_files[filename] = float(loss)

# Hyperparameters
sequence_length = 50
batch_size = 64
epochs = 25

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename in processed_files:
        print(f"Skipping {filename}: already trained.")
        continue

    file_path = os.path.join(directory, filename)

    # Load and preprocess the text
    text = load_text(file_path)
    input_sequences, total_words = create_sequences(text, sequence_length, sp)

    # Split into train and test
    X_train, X_test = train_test_split(input_sequences, test_size=0.2, random_state=42)

    # One-hot encode the sequences
    X_train = tf.keras.utils.to_categorical(X_train, num_classes=total_words)
    X_test = tf.keras.utils.to_categorical(X_test, num_classes=total_words)

    # Create the autoencoder
    model = create_autoencoder(sequence_length, total_words)

    # Train the autoencoder
    history = model.fit(
        X_train, X_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, X_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ]
    )

    # Record the normalized beginning and final validation loss
    beginning_val_loss = history.history['val_loss'][0]
    final_val_loss = history.history['val_loss'][-1]
    normalized_beginning_val_loss = beginning_val_loss / np.log(total_words)
    normalized_final_val_loss = final_val_loss / np.log(total_words)
    processed_files[filename] = (normalized_beginning_val_loss, normalized_final_val_loss)

    # Log the result for this file
    with open(trained_files_log, 'a') as f:
        f.write(f"{filename}: {normalized_beginning_val_loss}: {normalized_final_val_loss}\n")
    
    print(f"Processed {filename}: normalized beginning validation loss = {normalized_beginning_val_loss}, normalized final validation loss = {normalized_final_val_loss}")

    # Sort the results and write to results.csv
    sorted_results = sorted(processed_files.items(), key=lambda x: x[1][1])
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'normalized_beginning_val_loss', 'normalized_final_val_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for filename, (beginning_loss, final_loss) in sorted_results:
            writer.writerow({'filename': filename, 'normalized_beginning_val_loss': beginning_loss, 'normalized_final_val_loss': final_loss})