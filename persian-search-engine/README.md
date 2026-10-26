# Persian Search Engine — Hamshahri Corpus

A classical information-retrieval system for Persian news documents. The notebook implements document preprocessing and indexing, four ranking approaches, and evaluation against relevance judgements.

## Methods

- **Preprocessing:** normalize Arabic Yeh/Kaf variants, remove English letters and non-Persian symbols, tokenize, remove Persian stopwords, and retain sorted token and unique-term lists.
- **Indexing:** vocabulary, inverted index/postings, term frequency (TF), document frequency (DF), and logarithmic IDF.
- **BM25:** Okapi BM25 with `k1=1.2`, `b=0.75`, and `k3=1.5`.
- **Language model:** query-likelihood scoring with Dirichlet prior smoothing (`μ=2000` by default). Jelinek–Mercer smoothing is also supported in the class.
- **Vector space model:** TF–IDF vectors ranked by cosine similarity.
- **Hybrid:** weighted fusion of BM25, language-model, and TF–IDF scores (`0.5 / 0.3 / 0.2`). Scores are min–max normalized per query before fusion because the models produce values on different scales.
- **Evaluation:** macro-averaged Precision, Recall, and F1 across the supplied queries, using 20 α values in `[1, 10]`.

## Dataset setup

The dataset is intentionally not bundled in this repository. Obtain it from the course's authorized source and place it in this directory with the following structure:

```text
persian-search-engine/
├── README.md
├── requirements.txt
├── persian-search-engine.ipynb
└── HamshahriData/
    ├── HamshahriCorpus/
    │   ├── 2003/
    │   ├── 2004/
    │   ├── 2005/
    │   ├── 2006/
    │   └── 2007/
    ├── Queries/
    │   ├── 1.q
    │   └── ...
    ├── RelevanceAssessment/
    │   └── judgements.txt
    └── persian_stopwords.txt
```

The notebook expects document files (usually `.ham`) inside the year directories, query files ending in `.q`, and relevance judgements with one `query_id document_id` pair per line. Folder spelling must be `RelevanceAssessment`.

If your dataset is stored elsewhere, set `HAMSHAHRI_DATA_DIR` to the directory containing `HamshahriCorpus/`, `Queries/`, `RelevanceAssessment/`, and `persian_stopwords.txt` before launching Jupyter.

The notebook checks this layout at startup and reports missing paths if the dataset is absent or placed in a different folder. `HAMSHAHRI_DATA_DIR` can also be set to an absolute path, for example `/Users/yourname/datasets/HamshahriData`. Do not upload the corpus to GitHub; it is excluded by `.gitignore`.

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
jupyter notebook persian-search-engine.ipynb
```

Run notebook cells from top to bottom. Corpus preprocessing and evaluation can take time and require substantial memory depending on corpus size. The notebook's example query is `اقتصاد ایران`; change it to try another Persian query.

## Evaluation notes

The notebook evaluates all four models against the supplied query set and relevance judgements, then plots Precision, Recall, and F1 against α. Metrics are corpus- and judgement-dependent; rerun the notebook locally to regenerate them. Do not treat the narrative in an earlier exported course report as a guarantee of the results after code or dataset changes.

## Limitations and possible improvements

- The current preprocessing is intentionally lightweight and does not perform stemming or lemmatization.
- The hybrid model uses per-query min–max score normalization; this is simple and transparent, but other fusion strategies (e.g. rank fusion) could be compared.
- For a larger corpus, consider caching the preprocessed tokens/index, using sparse matrices, and adding MAP, MRR, or nDCG in addition to Precision/Recall/F1.
- The corpus is external and is not committed to GitHub. Keep dataset licensing and course policies in mind.

## Repository contents

- `persian-search-engine.ipynb` — implementation, evaluation, plots, and explanations.
- `requirements.txt` — Python dependencies.
