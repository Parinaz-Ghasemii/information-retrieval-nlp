# Information Retrieval & NLP Portfolio

A collection of coursework projects exploring information retrieval, text classification, graph-based ranking, and neural text generation. The projects move from classical sparse-vector methods to probabilistic retrieval models and sequence-to-sequence deep learning, with an emphasis on practical implementation, evaluation, and reproducibility.
 
**Focus areas:** Information Retrieval · Natural Language Processing · Text Mining · Deep Learning

## Projects

### 1. News Topic Classification with TF–IDF

**Folder:** [`news-classification-tfidf/`](news-classification-tfidf/)

A supervised text-classification project that represents news articles with TF–IDF features and compares linear classifiers on five topics: Politics, Sport, Technology, Entertainment, and Business.

**Methods and evaluation**
- TF–IDF vectorization with unigram features and unigram-plus-bigram features.
- Linear Support Vector Machine and one-vs-rest Logistic Regression.
- Stratified train/validation/test split (70% / 15% / 15%) and validation-based hyperparameter selection.
- Evaluation using accuracy, macro-precision, macro-recall, macro-F1, per-class scores, and a confusion matrix.
- An ablation comparison of unigram and bigram features.

The assignment specifies a dataset of 2,225 labelled examples. See the project folder for the notebook, implementation, and detailed results.

### 2. Persian Search Engine — Hamshahri Corpus

**Folder:** [`persian-search-engine/`](persian-search-engine/)

A Persian-language news retrieval system built around the Hamshahri corpus. It includes text normalization and stopword removal, vocabulary and inverted-index construction, term statistics, multiple ranking functions, and relevance-based evaluation.

**Methods**
- Persian text normalization, tokenization, and stopword filtering.
- Vocabulary, inverted index, term frequency (TF), document frequency (DF), and inverse document frequency (IDF).
- **Okapi BM25** for probabilistic term weighting and document-length normalization.
- **Query Language Model** with smoothing to address terms absent from an individual document.
- **Vector Space Model** using TF–IDF and cosine similarity.
- **Hybrid ranking** combining the BM25, language-model, and vector-space scores.
- Precision, recall, and F1 evaluation over 20 retrieval cutoffs controlled by \(\alpha\), from 1 to 10, using the provided queries and relevance judgements.

The notebook's saved output records a vocabulary of 49,853 unique terms and includes example rankings and metric plots. The project README documents its saved results, parameter choices, data layout, and known path setup checks.

### 3. PageRank from Scratch

**Folder:** [`pagerank/`](pagerank/)

An implementation of PageRank that demonstrates how link structure can represent a page's importance beyond raw in-degree counts.

**Methods**
- Construction of a column-stochastic transition matrix.
- Handling dangling nodes by distributing their transition probability uniformly across all nodes.
- Power iteration with a damping factor and convergence tolerance.
- Comparison of PageRank scores with in-degree and checks on probability normalization.

The notebook uses a generated graph, so it does not require an external graph dataset. See the PageRank README for implementation details and execution instructions.

### 4. Sherlock Holmes Text Generation with LSTM

**Folder:** [`sherlock-text-generation-lstm/`](sherlock-text-generation-lstm/)

A stylistic text-continuation experiment using a sequence-to-sequence encoder–decoder LSTM trained on Sherlock Holmes text. The encoder reads a source segment, and the decoder learns to generate a continuation.

**Methods**
- Text collection and preprocessing, tokenization, and vocabulary construction with special tokens such as `<PAD>`, `<SOS>`, `<EOS>`, and `<UNK>`.
- A PyTorch encoder–decoder LSTM architecture.
- Training and validation with a token-level cross-entropy objective.
- Evaluation using perplexity and Distinct-1 / Distinct-2 diversity metrics.
- Text generation using greedy decoding or sampling-based strategies, with controls for generation randomness.

The source corpus and generated training artifacts are not bundled. Refer to the project's README for its expected data layout, setup, commands, and notes on reproducing results.

## Repository Structure

```text
information-retrieval-nlp/
├── README.md
├── news-classification-tfidf/
├── persian-search-engine/
├── pagerank/
└── sherlock-text-generation-lstm/
```

Each project is kept in its own folder and should have a project-level README describing its implementation, dependencies, execution instructions, and results.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/information-retrieval-nlp.git
   cd information-retrieval-nlp
   ```

2. Open the project folder you want to explore.

3. Follow that project's `README.md` and install its dependencies. Dependency requirements differ between projects, so there is intentionally no single repository-wide environment.

4. For notebook-based work, activate the relevant environment and launch Jupyter:

   ```bash
   jupyter notebook
   ```

   Run the notebook cells in order after placing any required data in the location documented by that project's README.

## Data, Results, and Reproducibility

- External course datasets are not included by default. In particular, keep the Hamshahri corpus and the Sherlock Holmes raw-text collection out of Git unless redistribution is permitted.
- Model checkpoints, cached tensors, virtual environments, local secrets, and notebook checkpoint directories should not be committed.
- Results can depend on the exact dataset files, preprocessing choices, random seed, library versions, and training configuration. Consult each project README to distinguish saved notebook outputs from results freshly reproduced on your machine.
- The HW2 README includes caveats about comparing example rankings and interpreting the saved evaluation plots. The Sherlock project README documents that fresh training metrics require the raw corpus and a new run.

## Skills Demonstrated

- Text cleaning, tokenization, feature engineering, and stopword handling.
- Sparse text representations with TF–IDF and linear classification.
- Information retrieval structures: vocabularies, inverted indexes, postings, TF, DF, and IDF.
- Ranking approaches: BM25, smoothed query likelihood, cosine similarity, and score fusion.
- Relevance evaluation using precision, recall, F1, and confusion-matrix analysis for classification.
- Graph algorithms, stochastic matrices, convergence, and PageRank.
- PyTorch sequence modeling, encoder–decoder LSTMs, language-model evaluation, and text generation.

---

This repository is a learning portfolio of selected academic projects. See each project folder for the code, explanations, and project-specific results.
