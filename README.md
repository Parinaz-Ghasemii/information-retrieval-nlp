# Information Retrieval & NLP Projects

A portfolio of Information Retrieval and NLP projects from a B.Sc. in Computer Science, covering classical retrieval, Persian-language search, graph-based ranking, and neural text generation.

## Projects

- **HW1 — TF–IDF news classification:** add your completed project folder as `news-classification-tfidf/` (not included in this archive).
- [`persian-search-engine/`](persian-search-engine/) — Persian news retrieval over the Hamshahri corpus using BM25, a smoothed query language model, TF–IDF cosine similarity, and a weighted hybrid ranker.
- [`pagerank/`](pagerank/) — a PageRank implementation using a stochastic transition matrix, dangling-node handling, and power iteration.
- [`sherlock-text-generation-lstm/`](sherlock-text-generation-lstm/) — an encoder–decoder LSTM for Sherlock Holmes-style text continuation. The original submitted notebook and Python code are preserved unchanged; the report is documented in that project's README.

## HW3 report summary

### PageRank

On the notebook's generated 30-node graph, power iteration converged in 18 iterations with damping factor `0.85`. The transition matrix's columns sum to one, and the PageRank vector sums to one. The accompanying README answers the four assignment analysis questions, including dangling-node probability leakage, damping-factor sensitivity, link quality versus count, and teleportation for pages without incoming links.

### Sherlock LSTM

The output saved in the original submitted notebook reports a 656,465-token corpus, a 10,000-entry vocabulary, and three training epochs. Validation loss decreased from `4.2048` at epoch 1 to `1.9131` at epoch 3. Recorded evaluation metrics are validation perplexity `6.3415`, Distinct-1 `0.9446`, and Distinct-2 `1.0000`. Qualitative inspection of the saved generated text shows substantial word-level variety but limited sentence coherence in the displayed sample, so the metrics should not be interpreted as proof of successful style imitation. Detailed results and limitations are in [`sherlock-text-generation-lstm/README.md`](sherlock-text-generation-lstm/README.md).

These are the outputs embedded in the submitted notebooks, not new experiments performed during packaging. The Sherlock raw corpus and trained checkpoints are not included; regenerate results after supplying the corpus and running the original workflow.

## Reproducibility and data

Each project has its own README and dependency file. Course datasets, raw corpora, checkpoints, cached tensors, and local notebook checkpoints are not bundled. Check data availability and licensing before adding any source files to GitHub.

## Suggested repository layout

```text
information-retrieval-nlp/
├── README.md
├── news-classification-tfidf/       # add your existing HW1
├── persian-search-engine/
├── pagerank/
└── sherlock-text-generation-lstm/
```
