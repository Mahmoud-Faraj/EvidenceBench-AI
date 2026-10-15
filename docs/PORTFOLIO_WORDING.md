# Recruiter-facing project wording

Use these statements only with a link to this repository.

## Résumé

**EvidenceBench AI — Evaluation-First RAG Reliability Platform** | Python, FastAPI, Streamlit, scikit-learn,
Docker, GitHub Actions

- Built a tested retrieval and grounded-answering platform with BM25, TF-IDF/hybrid search, evidence citations,
  claim-support checks, and confidence-based abstention exposed through REST and interactive interfaces.
- Created a checksum-verified SciFact evaluation pipeline covering 5,183 abstracts and 188 labeled claims;
  achieved 0.794 MRR, 0.822 nDCG@10, and 0.924 Recall@10 with the strongest implemented BM25 baseline.
- Added reproducible experiments, bilingual English/Arabic processing, automated tests, containerization, and CI;
  documented limitations to distinguish measured performance from planned neural-retrieval work.

## LinkedIn project title

EvidenceBench AI — Evaluation-First RAG Reliability Platform

## LinkedIn description

Built an evaluation-first retrieval and grounded-answering platform that makes reliability visible. The project
implements BM25, TF-IDF and hybrid retrieval, evidence-constrained answers with citations, claim-support checks,
and confidence-based abstention. It includes a FastAPI service, Streamlit dashboard, Docker setup, automated
tests and CI. On the licensed SciFact development benchmark (5,183 abstracts and 188 labeled claims), the
strongest implemented baseline achieved 0.794 MRR, 0.822 nDCG@10 and 0.924 Recall@10. The repository includes
reproducible data preparation, machine-readable results, experiment notes and explicit limitations.

## Interview explanation

“I built EvidenceBench because many RAG demos focus on fluent output but do not expose whether the evidence was
retrieved correctly. I separated retrieval, answering and verification, added abstention, and benchmarked three
retrieval approaches on SciFact. An important finding was that equal-weight hybrid fusion underperformed BM25,
which gave me a concrete next experiment: add a genuinely semantic dense retriever and tune fusion without
leaking evaluation data.”
