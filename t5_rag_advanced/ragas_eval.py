"""
RAGAS evaluation for the Tiguan RAG pipeline.

Metrics:
  - Faithfulness:       Does the answer contain only info from the retrieved context?
  - ResponseRelevancy:  Is the answer relevant to the question?
  - LLMContextRecall:   Does the context contain everything needed to answer?
  - FactualCorrectness: Does the answer match the reference answer? (needs reference)

Run:
    python -m t5_rag_advanced.ragas_eval
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextRecall, FactualCorrectness

from commons.constants import ANTHROPIC_API_KEY
from t5_rag_advanced.embeddings.embeddings_client import EmbeddingsClient
from t5_rag_advanced.embeddings.text_processor import TextProcessor, SearchMode

# ── Golden test set ────────────────────────────────────────────────────────────
# (question, reference_answer)
# Reference answers are short, verifiable facts from the Tiguan manual.
GOLDEN_SET = [
    (
        "What is the recommended tire pressure for the Tiguan?",
        "The recommended cold tire inflation pressure for the Tiguan is listed in the tire pressure table in the manual and on the label inside the driver's door. Pressures vary by tire size and are given in PSI, kPa, and bar. Always check and adjust tire pressure when the tires are cold.",
    ),
    (
        "How do I activate the lane assist system?",
        "Lane Assist is switched on or off by selecting Lane Assist via the buttons on the multifunction steering wheel or through the Driver Assistance settings menu. When active, the system vibrates the steering wheel if you drift onto a lane marking. You can override it at any time by steering yourself.",
    ),
    (
        "What should I do if the engine warning light comes on?",
        "If the engine oil pressure warning light blinks red, stop the vehicle safely, stop the engine, and check the engine oil level. If the engine management warning light turns on, reduce speed and avoid heavy loads; visit an authorized Volkswagen dealer as soon as possible. Do not ignore a red warning light.",
    ),
    (
        "How do I pair my phone via Bluetooth?",
        "To pair a phone via Bluetooth, ensure your phone's Bluetooth is enabled. In the infotainment system, open the Phone menu or Radio/Media menu and select the option to pair a new device. Follow the on-screen instructions. Your phone must support Bluetooth version 2.1 or newer. If pairing fails via Phone or Radio/Media, try pairing directly from your phone's Bluetooth settings.",
    ),
    (
        "What is the maximum towing capacity of the Tiguan?",
        "I don't have information about that in the manual.",
    ),
    # Out-of-scope — answer should say context not found
    (
        "What is the capital of France?",
        "I don't have information about that in the manual.",
    ),
]

DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'database': 'vectordb',
    'user': 'postgres',
    'password': 'postgres',
}

SYSTEM_PROMPT = """
You are a Tiguan manual assistant. Answer ONLY using the provided RAG CONTEXT.
If the answer is not in the context, reply: "I don't have information about that in the manual."
Be concise.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}

##USER QUESTION:
{query}"""


def build_rag_pipeline():
    embeddings_client = EmbeddingsClient(
        endpoint="local",
        model_name="all-MiniLM-L6-v2",
        api_key="local",
    )
    text_processor = TextProcessor(embeddings_client, DB_CONFIG)
    llm = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return text_processor, llm


def run_rag(text_processor: TextProcessor, llm, question: str) -> tuple[str, list[str]]:
    """Returns (answer, retrieved_chunks)"""
    chunks = text_processor.search(
        search_mode=SearchMode.COSINE_DISTANCE,
        query=question,
        top_k=5,
        min_score=0.6,
        dimensions=384,
    )
    context = "\n\n".join(chunks) if chunks else "No relevant context found."
    augmented = USER_PROMPT.format(context=context, query=question)

    response = llm.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": augmented}],
    )
    return response.content[0].text, chunks


def main():
    print("Building RAG pipeline...")
    text_processor, llm = build_rag_pipeline()

    print(f"Running {len(GOLDEN_SET)} test cases...\n")
    samples = []
    for question, reference in GOLDEN_SET:
        print(f"  Q: {question}")
        answer, contexts = run_rag(text_processor, llm, question)
        print(f"  A: {answer[:120]}...\n")
        samples.append(SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts if contexts else ["No context retrieved."],
            response=answer,
            reference=reference,
        ))

    dataset = EvaluationDataset(samples=samples)

    # RAGAS needs an LLM judge — use Claude via LangChain wrapper
    judge_llm = LangchainLLMWrapper(
        ChatAnthropic(model="claude-sonnet-4-6", api_key=ANTHROPIC_API_KEY)
    )
    # Use the same local model as the RAG pipeline to avoid needing an OpenAI key
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    print("Running RAGAS evaluation...\n")
    results = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=judge_llm),
            ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings),
            LLMContextRecall(llm=judge_llm),
            FactualCorrectness(llm=judge_llm),
        ],
        embeddings=judge_embeddings,
    )

    print("\n=== RAGAS Results ===")
    print(results)
    df = results.to_pandas()
    print("\nPer-question breakdown:")
    score_cols = [c for c in df.columns if c not in ("retrieved_contexts", "response", "reference")]
    print(df[score_cols].to_string())

    base = Path(__file__).parent
    history_path = base / "ragas_history.json"
    out_path = base / "ragas_results.txt"

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metric_cols = [c for c in score_cols if c != "user_input"]
    new_entry = {
        "run": run_time,
        **{col: round(float(df[col].mean()), 4) for col in metric_cols},
    }

    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    history.append(new_entry)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    with out_path.open("w", encoding="utf-8") as f:
        # ── Comparison table ────────────────────────────────────────────
        f.write("RAGAS Evaluation — Historical Comparison\n")
        f.write("=========================================\n\n")
        col_w = 14
        header = f"{'Run':<22}" + "".join(f"{c[:col_w]:>{col_w}}" for c in metric_cols)
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for entry in history:
            row = f"{entry['run']:<22}" + "".join(f"{entry[c]:>{col_w}.4f}" for c in metric_cols)
            f.write(row + "\n")
        f.write("\n")

        # ── Latest run detail ────────────────────────────────────────────
        f.write(f"=== Latest Run: {run_time} ===\n")
        f.write(f"Model:      claude-sonnet-4-6\n")
        f.write(f"Embeddings: all-MiniLM-L6-v2\n")
        f.write(f"Questions:  {len(samples)}\n\n")
        f.write("Aggregate scores:\n")
        for col in metric_cols:
            f.write(f"  {col:<40} {df[col].mean():.4f}\n")
        f.write("\nPer-question breakdown:\n")
        f.write(df[score_cols].to_string())
        f.write("\n")

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
