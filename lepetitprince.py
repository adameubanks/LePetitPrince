import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sentencepiece as spm

# Load text with consistent preprocessing
def load_and_preprocess_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = text.lower()  # Normalize case
    text = ''.join(c for c in text if c.isalnum() or c.isspace())  # Remove punctuation
    return text

# Create sequences using SentencePiece tokenizer
def create_sequences(text, sequence_length, spm_model):
    token_list = spm_model.EncodeAsIds(text)
    input_sequences = []
    for i in range(sequence_length, len(token_list)):
        seq = token_list[i-sequence_length:i]
        input_sequences.append(seq)
    input_sequences = np.array(input_sequences)
    return input_sequences, len(spm_model)

# Create autoencoder
def create_autoencoder(sequence_length, total_words):
    model = tf.keras.models.Sequential()

    # Encoder
    model.add(tf.keras.layers.Embedding(total_words, 100, input_length=sequence_length))
    model.add(tf.keras.layers.LSTM(100, return_sequences=False))

    # Decoder
    model.add(tf.keras.layers.RepeatVector(sequence_length))
    model.add(tf.keras.layers.LSTM(100, return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(total_words, activation='softmax')))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Directory containing .txt files
directory = 'txt'

# File to save results
results_file = 'results.txt'

# Initialize SentencePiece model for tokenization
spm.SentencePieceTrainer.train(input=os.path.join(directory, '*.txt'), model_prefix='spm', vocab_size=8000)
sp = spm.SentencePieceProcessor(model_file='spm.model')

# Initialize results
results = []

# Hyperparameters
sequence_length = 50
batch_size = 64
epochs = 10

# Loop through each file in the directory
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)

    # Load and preprocess the text
    text = load_and_preprocess_text(file_path)
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

    # Record the normalized validation loss
    final_val_loss = history.history['val_loss'][-1]
    normalized_val_loss = final_val_loss / np.log(total_words)
    results.append((filename, normalized_val_loss))

    # Save the result for this file
    with open(results_file, 'a') as f:
        f.write(f"{filename}: {normalized_val_loss}\n")

# Sort results by normalized loss
sorted_results = sorted(results, key=lambda x: x[1])

# Save the sorted results
with open(results_file, 'a') as f:
    f.write("\nSorted Results (Least Loss First):\n")
    for filename, loss in sorted_results:
        f.write(f"{filename}: {loss}\n")

print("Processing complete. Results saved to", results_file)
