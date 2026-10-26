# Sherlock Holmes Text Continuation with an Encoder–Decoder LSTM

This project explores stylistic text continuation using a sequence-to-sequence (Seq2Seq) LSTM trained on text from the Sherlock Holmes collection by Arthur Conan Doyle. The encoder reads a context segment and passes its hidden and cell states to the decoder, which predicts the continuation.

> **Notebook integrity:** `sherlock_project.ipynb` is the original notebook from the submitted HW3 archive. Its code and saved outputs have been preserved unchanged. The original `src/` and `scripts/` Python files are also copied unchanged. The README, `.gitignore`, and `requirements.txt` were added for repository presentation and setup; no code cells were edited to create this package.

## Project structure

```text
sherlock-text-generation-lstm/
├── README.md
├── requirements.txt
├── sherlock_project.ipynb
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
├── scripts/
│   ├── evaluation.py
│   ├── generate.py
│   └── main.py
├── data/
│   └── raw/sherlock_holmes/  # put the source .txt files here
├── checkpoints/              # generated during training
├── logs/                     # generated during preprocessing/training
└── results/                  # generated during evaluation/generation
```

## Method

### Data preparation

The preprocessing pipeline combines `.txt` files from the Sherlock Holmes corpus, lowercases and cleans the text while retaining common punctuation, tokenizes on whitespace, and builds a token-to-ID vocabulary. The recorded run reports **656,465 tokens** and a vocabulary of **10,000 entries**. Special tokens are reserved for padding, sequence start, sequence end, and unknown words.

### Model and training configuration

The notebook configures an encoder–decoder architecture with a shared 128-dimensional embedding, two LSTM layers, 256 hidden units, and dropout of 0.5. It uses sequence chunks of length 100, with the first half used as the encoder input and the second half as the continuation target. The recorded experiment, `sherlock_lstm_light_v1`, uses Adam with learning rate 0.001, batch size 32, and 3 epochs.

## Report: results recorded in the submitted notebook

The following values are taken from the notebook's saved outputs. They document the submitted experiment and are **not results of a new run performed while packaging this repository**.

### Training loss

| Epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | 5.8384 | 4.2048 |
| 2 | 3.3870 | 2.6641 |
| 3 | 2.3328 | 1.9131 |

Both training and validation losses decreased over the three recorded epochs. Within this run, validation loss did not turn upward, so the saved three-epoch log does not by itself show validation-loss divergence. The notebook's own tuning note says longer runs had produced perplexity below 5 in other configurations and were regarded by the author as signs of memorization; that observation is part of the submitted notebook, not independently re-tested here.

### Quantitative evaluation

| Metric | Recorded value | Interpretation |
|---|---:|---|
| Validation perplexity | **6.3415** | Lower is better; this is below the assignment's stated underfitting warning threshold of 50, but perplexity alone does not establish stylistic quality. |
| Distinct-1 | **0.9446** | High unigram diversity in the generated sample. |
| Distinct-2 | **1.0000** | All bigrams in the evaluated sample were distinct under the metric used. This can reflect a short, varied sample and is not proof of coherence. |

### Qualitative assessment

The notebook includes generated continuations for prompts such as “It was a dark and stormy night” and “Sherlock Holmes sat in his chair.” In the saved first sample, the output begins with the prompt but continues with a sequence of mostly disconnected words (for example, “take let mind small must passed…”). This suggests that, despite the recorded perplexity and diversity scores, the model's generated text in these examples does **not yet demonstrate strong sentence-level coherence or a convincing Sherlock Holmes style**.

The results illustrate why language-generation evaluation needs both quantitative metrics and qualitative inspection: high Distinct-N values can coexist with weak grammar and narrative consistency. Potential next experiments would compare more data, sequence construction, training duration, and decoding settings, but the original code has intentionally not been modified in this portfolio copy.

## Dataset and running the original notebook

The source corpus is not included. Obtain the Sherlock Holmes `.txt` collection separately; place the files under `data/raw/sherlock_holmes/` when using the script-based project layout.

The original notebook is **Google Colab-specific** and expects a file named `sherlock_project.zip` at `/content/drive/MyDrive/sherlock_project.zip`. To execute it without editing its code:

1. From this folder, create a ZIP of the *contents* of this project folder (so `src/`, `scripts/`, and `requirements.txt` appear at the ZIP root), and name it `sherlock_project.zip`.
2. Upload that ZIP to the root of your Google Drive (`MyDrive`).
3. Open `sherlock_project.ipynb` in Google Colab, connect Drive, and run the cells in order.

For local use, the included Python modules are the original versions from the submitted archive; their paths and assumptions have not been refactored. The notebook's embedded saved outputs are useful as a record, but fresh metrics and generated samples should be produced after running with the full corpus.

## Dependencies

The added `requirements.txt` lists the libraries referenced by the project: PyTorch, NumPy, NLTK, and tqdm. PyTorch installation may differ depending on whether the runtime uses CPU, CUDA, or Apple Silicon.

## Data and generated files

Raw text, processed tensors/vocabulary, checkpoints, logs, and result files are excluded from Git because they are dataset-specific or reproducible outputs. Add only data you are permitted to redistribute. The empty placeholder directories show where generated files belong.
