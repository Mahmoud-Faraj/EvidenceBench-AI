# SciFact retrieval experiment

## Objective

Measure whether the implemented retrieval methods return evidence-bearing scientific abstracts for labeled
claims. This experiment evaluates retrieval only; it does not claim that generated answers are factually correct.

## Data

- 5,183 scientific abstracts from the official SciFact corpus
- 188 development claims with one or more labeled evidence documents
- Claims/annotations: CC BY 4.0
- Abstracts derived from S2ORC: ODC-By 1.0

The downloader verifies the official archive's SHA-256 checksum before preparation. Dataset files are not
redistributed in this repository.

## Compared systems

1. **BM25:** term-frequency ranking with document-length normalization.
2. **TF-IDF:** unigram/bigram sparse-vector cosine similarity.
3. **Hybrid:** equal-weight min-max-normalized fusion of BM25 and TF-IDF scores.

## Metrics

- **MRR:** rewards placing the first relevant document near rank one.
- **nDCG@10:** measures ranking quality in the first ten results with position discounting.
- **Recall@10:** measures the proportion of labeled evidence documents returned in the first ten results.

## Results

| Retriever | MRR | nDCG@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| BM25 | **0.794** | **0.822** | **0.924** |
| Hybrid BM25 + TF-IDF | 0.776 | 0.803 | 0.912 |
| TF-IDF | 0.724 | 0.757 | 0.887 |

## Interpretation

BM25 outperformed both TF-IDF and the equal-weight hybrid on every reported metric. The result is useful because
it rejects a convenient assumption: combining two retrievers does not guarantee improvement. TF-IDF and BM25
are correlated lexical methods, and min-max score fusion can weaken the stronger ranker.

## Reproduction

```bash
pip install -r requirements.txt
PYTHONPATH=src python scripts/download_scifact.py
PYTHONPATH=src python scripts/run_scifact_benchmark.py
```

## Next experiment

Add a dense multilingual sentence-transformer retriever, tune fusion weights on a training split, freeze the
configuration, and report final performance on the development split. Add bootstrap confidence intervals and
paired significance testing before making comparative research claims.
