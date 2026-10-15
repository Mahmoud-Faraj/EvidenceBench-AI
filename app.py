from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from evidencebench.data import load_documents, load_evaluation_set  # noqa: E402
from evidencebench.evaluation import evaluate_retriever  # noqa: E402
from evidencebench.pipeline import EvidencePipeline  # noqa: E402

st.set_page_config(page_title="EvidenceBench AI", page_icon="🔎", layout="wide")


@st.cache_resource
def build_pipeline() -> EvidencePipeline:
    return EvidencePipeline(load_documents(ROOT / "data" / "documents.json"))


pipeline = build_pipeline()
examples = load_evaluation_set(ROOT / "data" / "evaluation.json")
metrics = evaluate_retriever(pipeline.retriever, examples)

st.title("🔎 EvidenceBench AI")
st.caption("Evaluation-first retrieval, grounded answering, citation checking, and abstention")

metric_columns = st.columns(3)
metric_columns[0].metric("MRR", f"{metrics['mrr']:.3f}")
metric_columns[1].metric("Recall at 5", f"{metrics['recall_at_k']:.3f}")
metric_columns[2].metric("nDCG at 5", f"{metrics['ndcg_at_k']:.3f}")

question = st.text_input(
    "Ask in English or Arabic",
    value="What does retrieval augmented generation combine?",
)
top_k = st.slider("Retrieved passages", 1, 5, 3)

if st.button("Run evaluated answer", type="primary"):
    result = pipeline.answer(question, top_k=top_k)
    st.subheader("Answer")
    if result.answer.abstained:
        st.warning(result.answer.text)
    else:
        st.write(result.answer.text)

    columns = st.columns(3)
    columns[0].metric("Answer confidence", f"{result.answer.confidence:.2f}")
    columns[1].metric("Citation precision", f"{result.verification.citation_precision:.1%}")
    columns[2].metric("Supported claims", f"{result.verification.supported_claim_ratio:.1%}")

    st.subheader("Retrieval trace")
    trace = pd.DataFrame(
        {
            "document": [context.document.doc_id for context in result.contexts],
            "title": [context.document.title for context in result.contexts],
            "hybrid": [context.score for context in result.contexts],
            "lexical": [context.lexical_score for context in result.contexts],
            "semantic": [context.semantic_score for context in result.contexts],
        }
    )
    st.plotly_chart(
        px.bar(
            trace.melt(["document", "title"], var_name="score_type", value_name="score"),
            x="document",
            y="score",
            color="score_type",
            barmode="group",
            hover_data=["title"],
            title="Retrieval score components",
        ),
        width="stretch",
    )
    st.dataframe(trace, hide_index=True, width="stretch")

with st.expander("What this release demonstrates"):
    st.markdown(
        """
        - Hybrid lexical and semantic retrieval
        - English and Arabic tokenization
        - Evidence-constrained offline answer generation
        - Citation validation and claim-support scoring
        - Explicit abstention when retrieval confidence is insufficient
        - Reproducible ranking evaluation

        The offline generator is a transparent baseline. The architecture is designed so a transformer
        embedding model and an instruction-tuned LLM can be added and evaluated against the same tests.
        """
    )
