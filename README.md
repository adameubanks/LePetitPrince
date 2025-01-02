# Le Petit Prince: An AI Language Learning Experiment

## Overview
**What is the easiest language for an AI to learn?**

To explore this question, I fed 99 translations of *The Little Prince* into an autoencoder. This project aims to determine which languages an AI can "learn" and "understand" most effectively by analyzing the autoencoder performance across various languages and dialects.

## Why *The Little Prince*?

*The Little Prince* is one of the most translated works of fiction in the world. Its simple vocabulary and universal accessibility make it an ideal benchmark for comparing language processing across diverse languages.

## How Does the AI "Learn"?

Each translation was fed into an autoencoder, a type of neural network designed to compress and decompress input data. The autoencoder encodes the input into a compressed representation (bottleneck) and then reconstructs the original input. The idea is that translations that were easiest to compress and decompress represented the translations that the model understood "the best." 

The languages for which the autoencoder achieved:
- **Higher improvement during training** were easier for the model to learn.
- **Lower reconstruction error (loss)** were better understood by the model.

## Results
### Most Improved Training (Easiest to Learn):
1. **Chinese**
2. **Korean**
3. **Japanese** (Hiragana and Kanji)
4. **Bengali**
5. **Armenian**
6. **Toki Pona**
7. **Bulgarian**
8. **Buryat**

### Lowest Reconstruction Error (Best Understood):
1. **Hebrew**
2. **Amazigh (Berber)**
3. **Georgian**
4. **Korean**
5. **Lanna (Northern Thai)**
6. **Toki Pona**
7. **Karen**
8. **Bengali**

### Languages That Made Both Lists
1. **Korean**
2. **Toki Pona**
3. **Bengali**

## General Observations
- The AI performed best on languages with **denser writing systems**, such as Chinese and Japanese, where single characters can represent entire words.
- **Phonetic-based writing systems** like Korean and Bengali also showed strong results.
- **Toki Pona**, a constructed language designed for simplicity, appeared on both lists, demonstrating its accessibility for AI learning.
- High-density information languages, such as **Karen** and **Lanna**, excelled in reconstruction as well.

## Limitations
- The study was limited to a single text (*The Little Prince*) due to the challenge of finding works consistently translated into many diverse languages.
- Results were based on a single training iteration due to GPU constraints.
- Additional iterations, hyperparameter tuning, and larger datasets would improve the reliability of the results.
- Writing system differences (e.g., alphabetic vs. logographic) were not fully accounted for in this experiment.

## Conclusion
While this project doesn't definitively determine the "easiest language to learn," it highlights interesting trends:
- Languages with **efficient writing systems** or **high information density** performed better overall.
- Languages like **Chinese, Japanese, Korean, and Bengali** stand out as particularly "AI-friendly."

This experiment provided valuable insights into how AI processes different languages and laid the groundwork for more comprehensive future studies.

## Next Steps
- Expand the dataset with additional texts translated into many languages.
- Clean and preprocess data more thoroughly.
- Perform more training iterations with varied hyperparameters.
- Explore methods to account for differences in writing systems.

## Acknowledgments
- Un grand merci à [Petit Prince Collection](https://www.petit-prince-collection.com/index.php), qui a fourni toutes les traductions gratuitement.
- Thank you to Kaggle for providing the GPU hours needed to train.
- Thank you to chatGPT for help with the ideation and debugging of this project.