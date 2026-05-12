from pathlib import Path

import anthropic

from commons.constants import ANTHROPIC_API_KEY
from t5_rag_advanced.embeddings.embeddings_client import EmbeddingsClient
from t5_rag_advanced.embeddings.text_processor import TextProcessor, SearchMode

MANUALS = {
    "1": ("Microwave manual", Path(__file__).parent.parent / "t4_rag_fundamentals" / "microwave_manual.txt"),
    "2": ("Tiguan 2026 manual", Path(__file__).parent / "embeddings" / "tiguan-2026-owners-manual.pdf"),
}

DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'database': 'vectordb',
    'user': 'postgres',
    'password': 'postgres',
}

SYSTEM_PROMPT = """
You are a vehicle and appliance manual assistant powered by RAG (Retrieval-Augmented Generation).

Each user message contains two sections:
- ##RAG CONTEXT: relevant excerpts retrieved from the manual
- ##USER QUESTION: the user's actual question

Instructions:
- Answer ONLY using information found in the RAG CONTEXT.
- If the answer is not present in the RAG CONTEXT, reply: "I don't have information about that in the manual."
- Do not use outside knowledge. Do not answer questions unrelated to the loaded manual.

Formatting rules (always apply):
- Use a markdown heading (##) for the main topic.
- Use sub-headings (###) to group related information.
- Use bullet points or numbered steps for lists and procedures.
- Bold (**text**) key terms, button names, and warnings.
- Use > blockquotes for safety warnings or important notes.
- Keep each bullet concise — one idea per line.
- End with a brief summary or tip when relevant.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}

##USER QUESTION:
{query}"""


def main():
    embeddings_client = EmbeddingsClient(
        endpoint="local",
        model_name="all-MiniLM-L6-v2",
        api_key="local",
    )
    text_processor = TextProcessor(embeddings_client, DB_CONFIG)
    llm = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    print("Select manual:")
    for key, (name, _) in MANUALS.items():
        print(f"  {key}. {name}")
    choice = input("Choice (1/2): ").strip()
    manual_name, manual_path = MANUALS.get(choice, MANUALS["1"])

    load = input(f"Load '{manual_name}' into DB? (y/n): ").strip().lower()
    if load == "y":
        print(f"Processing {manual_path.name}...")
        text_processor.process_text_file(
            file_path=str(manual_path),
            chunk_size=300,
            overlap=40,
            dimensions=384,
            truncate=True,
        )

    print(f"\n{manual_name} Assistant (type 'exit' to quit)\n")
    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            break

        chunks = text_processor.search(
            search_mode=SearchMode.COSINE_DISTANCE,
            query=query,
            top_k=5,
            min_score=0.5,
            dimensions=384,
        )
        context = "\n\n".join(chunks) if chunks else "No relevant context found."

        augmented = USER_PROMPT.format(context=context, query=query)

        response = llm.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": augmented}],
        )
        print(f"\nAssistant: {response.content[0].text}\n")


main()
