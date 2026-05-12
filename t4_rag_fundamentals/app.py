import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from commons.constants import ANTHROPIC_API_KEY

_SYSTEM_PROMPT = """
You are a microwave oven assistant. Your role is to answer questions about the microwave manual.

Each user message contains two sections:
- ##RAG CONTEXT: relevant excerpts retrieved from the microwave manual
- ##USER QUESTION: the user's actual question

Instructions:
- Answer ONLY using information found in the RAG CONTEXT.
- If the answer is not present in the RAG CONTEXT, reply: "I don't have information about that in the microwave manual."
- Do not use any outside knowledge. Do not answer questions unrelated to the microwave manual.
- Be concise and precise.
"""

_USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION:
{query}"""

FAISS_INDEX = "microwave_faiss_index"
MANUAL_PATH = Path(__file__).parent / "microwave_manual.txt"


class MicrowaveRAG:

    def __init__(self, embeddings: HuggingFaceEmbeddings, llm_client: ChatAnthropic):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        print("Setting up vector store...")
        if os.path.exists(FAISS_INDEX):
            print(f"Loading existing FAISS index from '{FAISS_INDEX}'")
            return FAISS.load_local(FAISS_INDEX, self.embeddings, allow_dangerous_deserialization=True)
        print("No existing index found. Building new index...")
        return self._create_new_index()

    def _create_new_index(self) -> VectorStore:
        docs = TextLoader(str(MANUAL_PATH), encoding="utf-8").load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "."],
        )
        chunks = splitter.split_documents(docs)
        print(f"Split into {len(chunks)} chunks. Embedding...")
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        vectorstore.save_local(FAISS_INDEX)
        print(f"Index saved to '{FAISS_INDEX}'")
        return vectorstore

    def retrieve_context(self, query: str, k: int = 4, score: float = 0.3) -> str:
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=k, score_threshold=score
        )
        chunks = []
        for doc, relevance in results:
            print(f"  [score={relevance:.3f}] {doc.page_content[:80]}...")
            chunks.append(doc.page_content)
        return "\n\n".join(chunks)

    def augment_prompt(self, query: str, context: str) -> str:
        augmented = _USER_PROMPT.format(context=context, query=query)
        print(f"\n--- Augmented Prompt ---\n{augmented}\n---\n")
        return augmented

    def generate_answer(self, augmented_prompt: str) -> str:
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt),
        ]
        response = self.llm_client.invoke(messages)
        print(f"\nAssistant: {response.content}")
        return response.content


def main(rag: MicrowaveRAG):
    print("Microwave Manual Assistant (type 'exit' to quit)\n")
    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            break
        context = rag.retrieve_context(query)
        augmented = rag.augment_prompt(query, context)
        rag.generate_answer(augmented)


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm_client = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.0, api_key=ANTHROPIC_API_KEY)
main(MicrowaveRAG(embeddings, llm_client))
