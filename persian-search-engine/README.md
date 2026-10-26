# Persian Search Engine on the Hamshahri Corpus

An information-retrieval project for Persian news search using the Hamshahri corpus. The notebook implements a Persian text preprocessing pipeline, an inverted index, four ranking approaches, and relevance-based evaluation.

**Notebook:** [`persian-search-engine.ipynb`](persian-search-engine.ipynb)  
**Author:** Parinaz Ghasemi

> The Hamshahri corpus and its query/relevance files are not included in this repository. The result examples and evaluation plots below are the outputs already saved in the submitted notebook and report; they are not a newly rerun experiment.

## Project Overview

The system compares four ranking methods:

1. **Okapi BM25** — probabilistic ranking with term-frequency saturation and document-length normalization.
2. **Language Model** — query likelihood with Dirichlet prior smoothing. The implementation also contains an optional Jelinek–Mercer smoothing branch.
3. **Vector Space Model (VSM)** — TF–IDF document/query vectors ranked by cosine similarity.
4. **Hybrid Model** — combines normalized BM25, language-model, and VSM scores.

## Preprocessing and Indexing

The notebook performs these steps:

- Normalizes Arabic variants `ي` and `ك` to Persian `ی` and `ک`.
- Removes English letters and characters outside its Persian-letter/digit/whitespace allowlist.
- Tokenizes on whitespace and removes stopwords.
- Sorts each document's tokens and creates a sorted unique-term list.
- Builds a vocabulary, inverted index (term-to-document postings), term-frequency (TF), document-frequency (DF), and IDF values.

The IDF formula used is `log(N / DF(t))`, where `N` is the number of documents and `DF(t)` is the number of documents containing the term.

### Indexing output recorded in the notebook

| Item | Recorded output |
|---|---:|
| Vocabulary size | **49,853 unique terms** |
| Example document | `HAM2-820806-003` |
| Example postings-list term | `آزادراهی` |
| Example posting | `HAM2-850301-110` |

The vocabulary size is the value printed in the notebook's saved output. It depends on the exact corpus files and preprocessing setup.

## Ranking Models and Parameters

### BM25

The saved example uses `k1 = 1.2`, `b = 0.75`, and `k3 = 1.5`. These are conventional starting values: `k1` controls term-frequency saturation, `b` controls document-length normalization, and `k3` adjusts query-term-frequency influence.

### Language Model

The saved model setup uses **Dirichlet prior smoothing** with `mu = 2000`. Smoothing incorporates collection-level term probability so that a term absent from a document does not automatically make the entire query likelihood zero.

### Vector Space Model

Both document and query terms use TF–IDF weighting, and documents are sorted by cosine similarity to the query vector.

### Hybrid Model

The model min–max normalizes each model's scores for the query, then combines them with weights:

- BM25: `0.5`
- Language Model: `0.3`
- VSM: `0.2`

The weights are normalized in the implementation before combination.

## Code Results: Example Rankings

### BM25 — query `اقتصاد ایران`

Top 10 documents recorded in the notebook:

| Rank | Document ID | Score |
|---:|---|---:|
| 1 | `HAM2-830708-081` | 6.8845 |
| 2 | `HAM2-830216-126` | 6.8378 |
| 3 | `HAM2-811203-019` | 6.7979 |
| 4 | `HAM2-840412-145` | 6.7553 |
| 5 | `HAM2-851223-081` | 6.5938 |
| 6 | `HAM2-840217-124` | 6.5387 |
| 7 | `HAM2-851126-087` | 6.5201 |
| 8 | `HAM2-830624-086` | 6.5067 |
| 9 | `HAM2-851012-091` | 6.5067 |
| 10 | `HAM2-811214-032` | 6.4008 |

### Language Model — query `نفت ایران`

The notebook's printed example returns the following top 5. These scores are log-likelihood-style scores, so values are negative and a larger (less negative) score ranks higher.

| Rank | Document ID | Score |
|---:|---|---:|
| 1 | `HAM2-820814-028` | -9.1114 |
| 2 | `HAM2-850218-039` | -9.4937 |
| 3 | `HAM2-840330-069` | -9.7878 |
| 4 | `HAM2-831227-066` | -9.8365 |
| 5 | `HAM2-820329-103` | -9.8568 |

**Comparability note:** the Language Model example uses `نفت ایران`, while the BM25, VSM, and Hybrid examples below use `اقتصاد ایران`. In addition, the printed Language Model call does not pass the stopword set. Treat these as examples of each function's output, not as a controlled head-to-head comparison.

### VSM (TF–IDF cosine) — query `اقتصاد ایران`

| Rank | Document ID | Cosine score |
|---:|---|---:|
| 1 | `HAM2-811203-019` | 0.3305 |
| 2 | `HAM2-830216-126` | 0.3247 |
| 3 | `HAM2-840412-145` | 0.3123 |
| 4 | `HAM2-830708-081` | 0.3115 |
| 5 | `HAM2-851216-032` | 0.2777 |
| 6 | `HAM2-841206-149` | 0.2572 |
| 7 | `HAM2-840217-124` | 0.2501 |
| 8 | `HAM2-840614-063` | 0.2501 |
| 9 | `HAM2-811214-032` | 0.2468 |
| 10 | `HAM2-851012-091` | 0.2252 |

### Hybrid — query `اقتصاد ایران`

| Rank | Document ID | Hybrid score |
|---:|---|---:|
| 1 | `HAM2-811203-019` | 0.9937 |
| 2 | `HAM2-840412-145` | 0.9668 |
| 3 | `HAM2-830216-126` | 0.9519 |
| 4 | `HAM2-830708-081` | 0.9312 |
| 5 | `HAM2-840217-124` | 0.9071 |
| 6 | `HAM2-841206-149` | 0.8735 |
| 7 | `HAM2-851223-081` | 0.8647 |
| 8 | `HAM2-851012-091` | 0.8328 |
| 9 | `HAM2-811015-020` | 0.8295 |
| 10 | `HAM2-811214-032` | 0.8224 |

These rankings are corpus-specific examples saved in the notebook, not a claim that the listed documents are relevant to every query or that the scores from different models share the same scale.

## Evaluation Report

The assignment evaluation uses query files in `Queries/` and relevance labels in `RelevanceAssessment/judgements.txt`. For each query, let `n` be the number of judged relevant documents. At each of 20 values of `α` in the interval `[1, 10]`, the notebook sets the cutoff to `max(1, int(α × n))`, retrieves that many documents, and computes Precision, Recall, and F1. The metrics are macro-averaged across queries.

### Precision vs. α

![Precision vs alpha](results/precision_vs_alpha.png)

Precision falls as α grows: a larger cutoff includes more lower-ranked documents. In the saved plot, precision begins around **0.70–0.76** at α = 1 and falls to roughly **0.10** by α = 10. The curves become very close as α grows. The original report describes the Hybrid model as strongest at small cutoffs, with BM25 and the Language Model also competitive.

### Recall vs. α

![Recall vs alpha](results/recall_vs_alpha.png)

Recall rises as the cutoff grows because more judged relevant documents can be retrieved. In the saved plot, recall rises from roughly **0.70–0.76** at α = 1 to approximately **0.95–0.97** near α = 10. The Hybrid curve is highest over much of the range, while BM25 is competitive in the middle range.

### F1 vs. α

![F1 vs alpha](results/f1_vs_alpha.png)

F1 decreases as α increases in the saved plot. Precision drops as more documents are retrieved, and that reduction outweighs the recall gain. The curves are close, with the Hybrid model leading at low α and the methods converging as the cutoff becomes larger.

### Overall interpretation from the submitted report

- **BM25** provides a robust balance through term-frequency saturation and document-length normalization.
- **Language Model** remains competitive, with smoothing providing nonzero probabilities for terms not seen in a document.
- **TF–IDF VSM** is generally a little weaker at early cutoffs in the submitted analysis, although its recall approaches the other models as more results are included.
- **Hybrid** combines complementary scoring signals and performs best overall in the submitted report, especially for early precision/F1 and higher-cutoff recall.
- Smaller cutoffs (approximately α = 1–3) are described in the report as providing the best precision/recall balance.

**Result availability note:** the notebook contains the three plotted metric curves, but the exact macro-averaged metric arrays are not printed as a numeric table in the saved outputs. The ranges above are approximate readings from the saved charts, not exact table values. To publish exact averages, rerun the evaluation with the intended local dataset and export `bm25_res`, `lm_res`, `vsm_res`, and `hybrid_res` to a CSV or Markdown table.

## Dataset Setup

Place the dataset locally in this structure:

```text
HamshahriData/
├── HamshahriCorpus/
│   ├── 2003/
│   ├── 2004/
│   ├── 2005/
│   ├── 2006/
│   └── 2007/
├── Queries/
│   ├── 1.q
│   ├── 2.q
│   └── ...
├── RelevanceAssessment/
│   └── judgements.txt
└── persian_stopwords.txt
```

The corpus is organized by year. Each query file is named `<query-number>.q`. Each relevance-judgement line contains a query ID and a relevant document ID, for example:

```text
1 HAM2-821006-059
```

Do not commit the full corpus or other course-provided data unless you have permission to redistribute it.

## Installation and Execution

Use Python 3.9+ and install the notebook's external dependencies:

```bash
python -m pip install numpy matplotlib jupyter
jupyter notebook persian-search-engine.ipynb
```

The notebook defaults to `./HamshahriData` for `DATA_DIR`. You can set the environment variable before launching Jupyter:

```bash
export HAMSHAHRI_DATA_DIR="/absolute/path/to/HamshahriData"
jupyter notebook persian-search-engine.ipynb
```

### Two path checks before running

The uploaded notebook currently has two hard-coded path details worth checking on your machine:

1. `load_stopwords()` defaults to `HamshahriData/persian_stopwords.txt` rather than building the path from `DATA_DIR`. If your data folder is elsewhere, pass the correct path or change the default to `os.path.join(DATA_DIR, "persian_stopwords.txt")`.
2. The relevance-judgement cell refers to `RelativeAssesemnt/judgements.txt`; the dataset folder described in the assignment is `RelevanceAssessment/judgements.txt`. Update the spelling in that cell before running evaluation.

The notebook itself is included unchanged in this package; these notes document the current setup rather than silently modifying the code.

## Limitations and Next Steps

- Run the full evaluation against the precise corpus and relevance files you intend to use before presenting the curves as a fresh experiment.
- Export exact metric arrays to CSV and include them if you want a numerical result table in the repository.
- A controlled ranking comparison should use the same query text and stopword handling across all four rankers.
- The raw corpus is intentionally kept out of this repository package.
